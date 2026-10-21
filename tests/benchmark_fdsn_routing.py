"""Real downloads of erupt windows on a random sample of eruptions, with one way of choosing the data center:
- csv: the client of each eruptions.csv row (what scripts/download_windows.py does)
- federator: obspy's RoutingClient("earthscope-federator"), which asks EarthScope's fedcatalog which data center
  holds each station; the rows are deduplicated on (Network, Station) since the CSV client is ignored.

Not collected by pytest (network, minutes per eruption). Run one variant per process, e.g. in parallel:
    python -m tests.benchmark_fdsn_routing --variant csv --out csv.csv
    python -m tests.benchmark_fdsn_routing --variant federator --out federator.csv
"""

import argparse
import tempfile
import time

import pandas as pd
from obspy import read
from obspy.clients.fdsn import RoutingClient
from obspy.clients.fdsn.header import FDSNNoDataException

from scripts import download_windows as dw


def sample_eruptions(path, n, seed):
    df = pd.read_csv(f"{path}/eruptions.csv", index_col=0)
    keys = df[["Volcano Number", "Start Date"]].drop_duplicates().sort_values(["Volcano Number", "Start Date"])
    keys = keys.sample(n, random_state=seed)
    return [(num, start, df[(df["Volcano Number"] == num) & (df["Start Date"] == start)]) for num, start in keys.values]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["csv", "federator"], required=True)
    parser.add_argument("--path", default=dw.DATA, help="directory with eruptions.csv")
    parser.add_argument("--n", type=int, default=30, help="number of eruptions")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True, help="CSV with one row per eruption")
    args = parser.parse_args()

    if args.variant == "federator":
        router = RoutingClient("earthscope-federator")
        get_stations = router.get_stations
        def routed_get_stations(**kwargs):
            # fedcatalog answers "nothing matches" with a 404, not a 204: make it a 204 for operating_rows
            try:
                return get_stations(**kwargs)
            except Exception as exc:
                if "No data matched request" in str(exc):
                    raise FDSNNoDataException(str(exc))
                raise
        router.get_stations = routed_get_stations
        dw.get_client = lambda client_name: router

    calls = []
    fetch_station = dw.fetch_station
    def counted_fetch_station(client_name, network, station, start, end):
        calls.append((client_name, network, station))
        return fetch_station(client_name, network, station, start, end)
    dw.fetch_station = counted_fetch_station

    results = []
    with tempfile.TemporaryDirectory() as out_dir:
        for num, start, rows in sample_eruptions(args.path, args.n, args.seed):
            if args.variant == "federator":
                rows = rows.drop_duplicates(["Network", "Station"]).assign(**{"Client Name": "federator"})
            stem = dw.window_stem(num, rows.iloc[0]["Volcano Name"], start, "erupt")
            calls.clear()
            t0 = time.time()
            result = dw.fetch_window(rows, start, 12, 24, out_dir, stem)
            channels = ""
            if result == "saved":
                channels = ",".join(sorted(tr.id for tr in read(f"{out_dir}/{stem}.mseed")))
            results.append({"window": stem, "result": result, "waveform_calls": len(calls),
                            "seconds": round(time.time() - t0), "channels": channels})
            print(f"== {stem}: {result}, {len(calls)} calls {channels}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print(df["result"].value_counts().to_string(), f"\nwaveform calls: {df['waveform_calls'].sum()}",
          f"\nminutes: {df['seconds'].sum() / 60:.0f}")
