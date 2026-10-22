import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime, Stream
from obspy.clients.fdsn.header import FDSNNoDataException

from download_windows import get_client # script dir is on sys.path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

COLUMNS = ["volcano_number", "volcano_name", "start_date", "client", "network", "station", "location", "channel",
           "sampling_rate", "npts", "coverage", "status", "detail"]

# Tested 2026-10-22 on 3 eruptions per data center (plan 0010):
# - EARTHSCOPE: the GET answer is complete, but ObsPy's get_waveforms trims the Stream, and Stream.trim empties it
#   when it holds a 0 Hz log channel (ACE, OCF): 1 trace instead of 166 on 2010-05-21.
# - RESIF: GET with 100 stations is a 400 Bad request; bulk works up to 100 lines (248 is a 400).
# - GEOFON, INGV: same trace ids both ways.
REQUEST_FORM = {"EARTHSCOPE": "bulk", "GEOFON": "bulk", "RESIF": "bulk", "INGV": "bulk"}


def request(client_name, pairs, t0, t1):
    client = get_client(client_name)
    if REQUEST_FORM[client_name] == "bulk":
        return client.get_waveforms_bulk([(net, sta, "*", "*", t0, t1) for net, sta in pairs])
    # GET returns the net x sta cross product: keep the asked pairs only
    st = client.get_waveforms(network=",".join(sorted({n for n, _ in pairs})),
                              station=",".join(sorted({s for _, s in pairs})),
                              location="*", channel="*", starttime=t0, endtime=t1)
    asked = set(pairs)
    return Stream([tr for tr in st if (tr.stats.network, tr.stats.station) in asked])


def probe_batch(client_name, pairs, t0, minutes):
    """One row per channel with data, one row per station without. Never retried."""
    t1 = t0 + minutes * 60
    try:
        st = request(client_name, pairs, t0, t1)
    except FDSNNoDataException:
        return [dict(network=n, station=s, status="nodata") for n, s in pairs]
    except Exception as exc:
        detail = f"{type(exc).__name__}: {str(exc).split(chr(10))[0]}" # the rest can be a binary server answer
        return [dict(network=n, station=s, status="error", detail=detail) for n, s in pairs]

    rows = []
    for trace_id in sorted({tr.id for tr in st}):
        traces = [tr for tr in st if tr.id == trace_id]
        for tr in traces:
            tr.trim(t0, t1) # one by one: bulk answers aren't trimmed, and Stream.trim breaks on 0 Hz channels
        sr = traces[0].stats.sampling_rate
        npts = sum(tr.stats.npts for tr in traces)
        net, sta, loc, cha = trace_id.split(".")
        rows.append(dict(network=net, station=sta, location=loc, channel=cha, sampling_rate=sr, npts=npts,
                         coverage=npts / (sr * minutes * 60) if sr else None, status="data"))
    with_data = {(tr.stats.network, tr.stats.station) for tr in st}
    rows += [dict(network=n, station=s, status="nodata") for n, s in pairs if (n, s) not in with_data]
    return rows


def probe_unit(unit, minutes, batch):
    (volcano_number, start_date, client_name), rows = unit
    pairs = [tuple(p) for p in rows[["Network", "Station"]].drop_duplicates().values.tolist()]
    t0 = UTCDateTime(start_date)
    out = []
    for i in range(0, len(pairs), batch):
        out += probe_batch(client_name, pairs[i:i + batch], t0, minutes)
    info = dict(volcano_number=volcano_number, volcano_name=rows.iloc[0]["Volcano Name"],
                start_date=start_date, client=client_name)
    return pd.DataFrame([{**info, **r} for r in out], columns=COLUMNS), len(pairs)


#######################################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Which stations return which channels, per eruption: short get_waveforms probes.")
    parser.add_argument("--path", default=DATA, help="directory with eruptions.csv")
    parser.add_argument("--out", default=DATA / "channel_survey.csv", help="output CSV; units already in it are skipped")
    parser.add_argument("--limit", type=int, help="only the first N eruptions")
    parser.add_argument("--minutes", type=int, default=10, help="probe length from the start date 00:00 UTC")
    parser.add_argument("--batch", type=int, default=100, help="max stations per request")
    args = parser.parse_args()

    df_eruptions = pd.read_csv(f"{args.path}/eruptions.csv", index_col=0)
    # one group per eruption; Volcano Number, not Volcano Name (names are not unique)
    groups = list(df_eruptions.groupby(["Volcano Number", "Start Date"], sort=True))[:args.limit]

    done = set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out, usecols=["volcano_number", "start_date", "client"])
        done = set(prev.itertuples(index=False, name=None))

    # one queue of units per data center, in eruption order
    units = {}
    for (volcano_number, start_date), rows in groups:
        for client_name, g in rows.groupby("Client Name"):
            if (volcano_number, start_date, client_name) not in done:
                units.setdefault(client_name, []).append(((volcano_number, start_date, client_name), g))

    lock = threading.Lock()

    def run_client(client_units):
        # one thread per data center: each one gets a single request at a time
        for unit in client_units:
            t = time.time()
            res, asked = probe_unit(unit, args.minutes, args.batch)
            with lock:
                res.to_csv(args.out, mode="a", header=not os.path.exists(args.out), index=False)
                (volcano_number, start_date, client_name), rows = unit
                n_data = res[res.status == "data"].drop_duplicates(["network", "station"]).shape[0]
                print(f"{rows.iloc[0]['Volcano Name']} {start_date} {client_name}: "
                      f"{n_data}/{asked} stations with data, {time.time() - t:.1f} s", flush=True)

    print(f"{sum(len(u) for u in units.values())} units to probe, {len(done)} already done")
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in [pool.submit(run_client, u) for u in units.values()]:
            f.result()

    # stations (not channel rows) per status and data center, over the whole file
    key = ["volcano_number", "start_date", "client", "network", "station"]
    survey = pd.read_csv(args.out, usecols=key + ["status"]).drop_duplicates(key)
    print(pd.crosstab(survey.client, survey.status, margins=True))
