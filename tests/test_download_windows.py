"""Checks scripts/download_windows.py offline: a fake FDSN client stands in for obspy's Client."""

import runpy
import sys
from pathlib import Path

import numpy as np
import obspy.clients.fdsn
import pandas as pd
import pytest
from obspy import Inventory, Stream, Trace, UTCDateTime, read
from obspy.core.inventory import Network, Station
from obspy.clients.fdsn.header import FDSNNoDataException

from scripts import download_windows

SCRIPT = Path(download_windows.__file__)


class FakeFDSN:
    """Data centers in memory. `stations` maps (client, network, station) to the channels it serves as
    {code: sampling rate}; {} is a station that is operating but has no waveforms. A station missing from
    `stations` is not operating. Requests to a client in `down`, or for a station in `broken`, fail with a
    server error."""

    def __init__(self, stations=None, down=(), broken=()):
        self.stations = stations or {}
        self.down = set(down)
        self.broken = set(broken)
        self.waveform_calls = [] # (client, network, station, starttime)

    def client(self, name): # replaces obspy's Client(name)
        return FakeClient(self, name)


class FakeClient:
    def __init__(self, fdsn, name):
        self.fdsn, self.name = fdsn, name

    def get_stations(self, network, station, level, **kwargs):
        if self.name in self.fdsn.down:
            raise Exception("500 Internal Server Error")
        # net x sta cross product, like the real GET with comma lists
        nets, stas = network.split(","), station.split(",")
        found = [(n, s) for (c, n, s) in self.fdsn.stations if c == self.name and n in nets and s in stas]
        if not found:
            raise FDSNNoDataException("No data available for request.")
        return Inventory(networks=[Network(n, stations=[Station(s, 0, 0, 0)]) for n, s in found], source="fake")

    def get_waveforms(self, network, station, location, channel, starttime, endtime):
        key = (self.name, network, station)
        self.fdsn.waveform_calls.append((*key, starttime))
        if self.name in self.fdsn.down or key in self.fdsn.broken:
            raise Exception("500 Internal Server Error")
        channels = self.fdsn.stations.get(key)
        if not channels:
            raise FDSNNoDataException("No data available for request.")
        return Stream([Trace(np.zeros(100, dtype=np.int32),
                             header=dict(network=network, station=station, location="", channel=code,
                                         sampling_rate=rate, starttime=starttime))
                       for code, rate in channels.items()])


@pytest.fixture
def fake(monkeypatch):
    fdsn = FakeFDSN()
    monkeypatch.setattr(download_windows, "Client", fdsn.client)
    monkeypatch.setattr(download_windows, "_clients", {})
    return fdsn


def rows(*keys):
    """Candidate stations of one eruption, in CSV order."""
    return pd.DataFrame([{"Volcano Number": 100, "Volcano Name": "Alpha",
                          "Client Name": c, "Network": n, "Station": s} for c, n, s in keys])


def fetch(fake, keys, tmp_path):
    result = download_windows.fetch_window(rows(*keys), "2010-01-01", 12, 24, tmp_path, "w")
    ids = sorted(tr.id for tr in read(tmp_path / "w.mseed")) if result == "saved" else []
    return result, ids


def called(fake):
    return [s for _, _, s, _ in fake.waveform_calls]


ES, GF = "EARTHSCOPE", "GEOFON"

############################################ fetch_window ############################################

def test_slots_are_filled_from_several_stations(fake, tmp_path):
    fake.stations = {(ES, "XX", "S1"): {"EHZ": 100},
                     (ES, "XX", "S2"): {"EHZ": 100, "LHN": 1, "LHE": 1}}
    result, ids = fetch(fake, [(ES, "XX", "S1"), (ES, "XX", "S2")], tmp_path)
    # S2's EHZ is not kept: the HZ slot is already filled by S1
    assert (result, ids) == ("saved", ["XX.S1..EHZ", "XX.S2..LHE", "XX.S2..LHN"])
    assert (tmp_path / "w.xml").exists()


def test_hz_slot_prefers_ehz(fake, tmp_path):
    fake.stations = {(ES, "XX", "S1"): {"HHZ": 100, "BHZ": 40, "EHZ": 100}}
    assert fetch(fake, [(ES, "XX", "S1")], tmp_path)[1] == ["XX.S1..EHZ"]


def test_hz_below_20_hz_is_not_kept(fake, tmp_path):
    fake.stations = {(ES, "XX", "S1"): {"BHZ": 10},
                     (ES, "XX", "S2"): {"HHZ": 100}}
    assert fetch(fake, [(ES, "XX", "S1"), (ES, "XX", "S2")], tmp_path)[1] == ["XX.S2..HHZ"]


def test_stops_calling_once_every_slot_is_filled(fake, tmp_path):
    fake.stations = {(ES, "XX", "S1"): {"EHZ": 100, "LHN": 1, "LHE": 1},
                     (ES, "XX", "S2"): {"EHZ": 100}}
    fetch(fake, [(ES, "XX", "S1"), (ES, "XX", "S2")], tmp_path)
    assert called(fake) == ["S1"]


def test_stations_not_operating_are_not_called(fake, tmp_path):
    fake.stations = {(ES, "XX", "S2"): {"EHZ": 100}}
    result, _ = fetch(fake, [(ES, "XX", "S1"), (ES, "XX", "S2")], tmp_path)
    assert (result, called(fake)) == ("saved", ["S2"])


def test_at_most_max_tries_stations_are_called(fake, tmp_path):
    keys = [(ES, "XX", f"S{i:02d}") for i in range(download_windows.MAX_TRIES + 2)]
    fake.stations = {k: {} for k in keys}
    fake.stations[keys[-1]] = {"EHZ": 100} # data only past MAX_TRIES
    result, _ = fetch(fake, keys, tmp_path)
    assert (result, len(fake.waveform_calls)) == ("nodata", download_windows.MAX_TRIES)


def test_no_data_anywhere_is_nodata(fake, tmp_path):
    fake.stations = {(ES, "XX", "S1"): {}}
    assert fetch(fake, [(ES, "XX", "S1"), (ES, "XX", "S2")], tmp_path) == ("nodata", [])
    assert list(tmp_path.iterdir()) == []


def test_metadata_server_error_is_an_error(fake, tmp_path):
    fake.stations = {(ES, "XX", "S1"): {"EHZ": 100}}
    fake.down = {ES}
    assert fetch(fake, [(ES, "XX", "S1")], tmp_path) == ("error", [])


def test_waveform_server_error_is_an_error(fake, tmp_path):
    fake.stations = {(ES, "XX", "S1"): {"EHZ": 100}}
    fake.broken = {(ES, "XX", "S1")}
    assert fetch(fake, [(ES, "XX", "S1")], tmp_path) == ("error", [])


def test_data_from_another_client_beats_an_error(fake, tmp_path):
    fake.stations = {(GF, "XX", "S2"): {"EHZ": 100}}
    fake.down = {ES}
    assert fetch(fake, [(ES, "XX", "S1"), (GF, "XX", "S2")], tmp_path) == ("saved", ["XX.S2..EHZ"])


def test_same_station_on_two_clients(fake, tmp_path):
    # Telica 2009-11-04: NU.CNGN is operating on EARTHSCOPE but has no data there; GEOFON has the data
    fake.stations = {(ES, "NU", "CNGN"): {}, (GF, "NU", "CNGN"): {"BHZ": 20, "HHZ": 100}}
    result, ids = fetch(fake, [(ES, "NU", "CNGN"), (GF, "NU", "CNGN")], tmp_path)
    assert (result, ids) == ("saved", ["NU.CNGN..BHZ"])
    assert [c for c, _, _, _ in fake.waveform_calls] == [ES, GF]


def test_operating_rows_ignores_extra_stations_of_the_cross_product(fake):
    # asking for XX,YY x S1,S2 also returns XX.S2, which is not a candidate
    fake.stations = {(ES, "XX", "S1"): {}, (ES, "XX", "S2"): {}, (ES, "YY", "S2"): {}}
    kept, error = download_windows.operating_rows(rows((ES, "XX", "S1"), (ES, "YY", "S2")), None, None)
    assert (kept["Station"].tolist(), error) == (["S1", "S2"], False)

############################################ the whole script ############################################

# eruption: (start, end, stations); the gaps are 365 d, 191 d, 190 d and 1 d
ALPHA = [("2010-01-01", "2010-01-11", ["A1", "S0"]),
         ("2011-01-11", "2011-01-21", ["S0", "B1"]),
         ("2011-07-31", "2011-08-10", ["S0"]),
         ("2012-02-16", "2012-02-20", ["S0"]),
         ("2012-02-21", "2012-02-22", ["S0"])]
BETA = [("2015-06-01", "2015-06-05", ["S0"])]


def write_csv(path):
    records = []
    for number, name, eruptions in ((100, "Alpha", ALPHA), (200, "Beta", BETA)):
        for start, end, stations in eruptions:
            for sta in stations:
                client = GF if sta == "B1" else ES
                records.append({"Start Date": start, "End Date": end, "Volcano Name": name,
                                "Volcano Number": number, "Latitude": 0, "Longitude": 0,
                                "Client Name": client, "Network": "XX", "Station": sta})
    pd.DataFrame(records).to_csv(path / "eruptions.csv")


def run_script(fake, tmp_path, monkeypatch, skip=()):
    """Runs the script on ALPHA and BETA; returns the files it writes."""
    write_csv(tmp_path)
    argv = ["download_windows.py", "--path", str(tmp_path)]
    if skip:
        (tmp_path / "skip.txt").write_text("\n".join(skip))
        argv += ["--skip", str(tmp_path / "skip.txt")]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(obspy.clients.fdsn, "Client", fake.client)
    runpy.run_path(str(SCRIPT), run_name="__main__")
    return sorted(p.name for p in (tmp_path / "windows").iterdir())


def test_one_erupt_window_per_eruption_and_up_to_3_quiet_per_gap(fake, tmp_path, monkeypatch):
    # nothing is operating: every window gets a .nodata marker, which shows the dates the script tried
    files = run_script(fake, tmp_path, monkeypatch)
    assert files == sorted([
        *[f"100_Alpha_{start}_erupt.nodata" for start, _, _ in ALPHA],
        "200_Beta_2015-06-01_erupt.nodata",
        # 365 d gap: midpoint 2010-07-12 12:00 and both offsets
        "100_Alpha_2010-05-13_quiet.nodata", "100_Alpha_2010-07-12_quiet.nodata", "100_Alpha_2010-09-10_quiet.nodata",
        # 191 d gap: midpoint - 60 d starts exactly 35 d after the end of A; midpoint + 60 d is too close to B
        "100_Alpha_2011-02-25_quiet.nodata", "100_Alpha_2011-04-26_quiet.nodata",
        # 190 d gap: both offsets are too close, the midpoint is kept without the margin
        "100_Alpha_2011-11-13_quiet.nodata",
        # 1 d gap: no quiet window; Beta has a single eruption: no quiet window
    ])


def test_quiet_windows_use_the_stations_of_a_then_b(fake, tmp_path, monkeypatch):
    fake.stations = {(ES, "XX", "A1"): {}, (ES, "XX", "S0"): {}, (GF, "XX", "B1"): {}}
    run_script(fake, tmp_path, monkeypatch)
    start = UTCDateTime("2010-07-12T12:00:00") - 12 * 3600 # midpoint of the first gap
    assert [s for _, _, s, t in fake.waveform_calls if t == start] == ["A1", "S0", "B1"]


def test_skipped_windows_are_not_fetched(fake, tmp_path, monkeypatch):
    skip = ["100_Alpha_2010-07-12_quiet.mseed", "100_Alpha_2010-01-01_erupt.nodata"]
    files = run_script(fake, tmp_path, monkeypatch, skip)
    assert "100_Alpha_2010-07-12_quiet.nodata" not in files
    assert "100_Alpha_2010-01-01_erupt.nodata" not in files
    assert "100_Alpha_2010-05-13_quiet.nodata" in files
