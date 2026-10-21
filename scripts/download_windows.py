import argparse
import os
import pandas as pd
import re
import unicodedata
from pathlib import Path

from obspy import UTCDateTime, Stream, Inventory
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MAX_TRIES = 10 # max stations called per window
QUIET_SPACING_DAYS = 60 # distance between the quiet windows of the same gap
QUIET_MARGIN_DAYS = 35  # min distance between an extra quiet window and the end of A or the start of B

# channel slots kept per window: slot -> channel codes that can fill it, in order of preference
SLOTS = {
    "HZ": ["EHZ", "BHZ", "HHZ"], # high-rate vertical (1-9 Hz volcanic band), must be >= 20 Hz
    "LHN": ["LHN"],              # very-long-period horizontals (Jerk transient)
    "LHE": ["LHE"],
}
CHANNEL_LIST = ",".join(code for codes in SLOTS.values() for code in codes)

############# One window = the channels of several stations, combined into one mseed + one StationXML ################

def window_stem(volcano_number, volcano_name, start_date, tag):
    # same cleaning as in notebooks/get_seismo_data.py
    volcano_name_clean = re.sub(
        r"[^A-Za-z0-9-]",
        "",
        re.sub(
            r"\s+", "-",
            unicodedata.normalize("NFKD", volcano_name)
            .encode("ascii", "ignore")
            .decode("ascii")
            .strip()
        ),
    )
    return f"{volcano_number}_{volcano_name_clean}_{str(start_date)[:10]}_{tag}"


_clients = {}

def get_client(client_name):
    # one Client per data center: creating one costs a request
    if client_name not in _clients:
        _clients[client_name] = Client(client_name)
    return _clients[client_name]


def fetch_station(client_name, network, station, start, end):
    """Same requests as get_seismo_data, without writing files.
    Returns (st, inv, status): status is "ok", "nodata" (204) or "error" (anything else, e.g. server down)."""
    try:
        client = get_client(client_name)
        st = client.get_waveforms(network=network, station=station, location="*",
                                  channel=CHANNEL_LIST, starttime=start, endtime=end)
        inv = client.get_stations(network=network, station=station, level="response")
    except FDSNNoDataException:
        return None, None, "nodata"
    except Exception as exc:
        print(f"Error for {client_name} {network}.{station}: {exc}")
        return None, None, "error"
    print(f"Data from {client_name} {network}.{station}")
    return st, inv, "ok"


def operating_rows(rows, start, end):
    """(rows whose station has a CHANNEL_LIST channel operating during the window, in CSV order,
    True if a metadata request failed other than with 204)."""
    ok = set()
    error = False
    for client_name, g in rows.groupby("Client Name"):
        pairs = g[["Network", "Station"]].drop_duplicates().values.tolist()
        # GET with comma lists: RESIF rejects bulk POSTs of more than a few lines.
        # The net x sta cross product can return extra stations; the mask below drops them.
        for i in range(0, len(pairs), 200):
            chunk = pairs[i:i + 200]
            try:
                inv = get_client(client_name).get_stations(network=",".join(sorted({n for n, _ in chunk})),
                                          station=",".join(sorted({s for _, s in chunk})),
                                          channel=CHANNEL_LIST, starttime=start, endtime=end, level="station")
            except FDSNNoDataException:
                continue
            except Exception as exc:
                print(f"get_stations failed for {client_name}: {exc}")
                error = True
                continue
            ok |= {(client_name, net.code, sta.code) for net in inv for sta in net}
    mask = [tuple(k) in ok for k in rows[["Client Name", "Network", "Station"]].values]
    return rows[mask], error


def fetch_window(rows, start_date, t_before, t_after, out_dir, stem):
    """Returns "saved", "nodata" (every request answered, none with usable data) or "error"
    (no usable data, and at least one request failed other than with 204: worth retrying)."""
    d = UTCDateTime(start_date)
    start, end = d - t_before*3600, d + t_after*3600
    combined_st = Stream()
    combined_inv = Inventory(networks=[])
    filled = {} # slot -> trace id

    candidates, error = operating_rows(rows, start, end)
    for _, row in candidates.head(MAX_TRIES).iterrows():
        if len(filled) == len(SLOTS): # every slot filled: stop calling stations
            break

        st, inv, status = fetch_station(row['Client Name'], row['Network'], row['Station'], start, end)
        if status != "ok":
            error |= status == "error"
            continue

        kept = False
        for slot, codes in SLOTS.items():
            if slot in filled:
                continue
            for code in codes:
                traces = [tr for tr in st.select(channel=code)
                          if slot != "HZ" or tr.stats.sampling_rate >= 20]
                if traces:
                    trace_id = traces[0].id # first location code only, with all its gap pieces
                    combined_st += Stream([tr for tr in traces if tr.id == trace_id])
                    filled[slot] = trace_id
                    kept = True
                    break

        if kept:
            combined_inv += inv

    if len(combined_st) == 0:
        return "error" if error else "nodata"

    print(f"Saving {stem}: {filled}")
    combined_st.write(f"{out_dir}/{stem}.mseed", format="MSEED")
    combined_inv.write(f"{out_dir}/{stem}.xml", format="STATIONXML")
    return "saved"

#######################################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One combined erupt window per eruption and up to 3 quiet windows per gap.")
    parser.add_argument("--path", default=DATA, help="directory with eruptions.csv; windows are written to {path}/{prefix}/")
    parser.add_argument("--skip", help="text file with one existing filename per line; those windows are skipped")
    parser.add_argument("--prefix", default="windows", help="output sub-directory (and bucket prefix)")
    parser.add_argument("--bucket", help="GCS bucket: upload each window to gs://{bucket}/{prefix}/ as soon as it is saved, then delete the local copy")
    args = parser.parse_args()

    bucket = None
    if args.bucket:
        from google.cloud import storage # only needed when uploading
        bucket = storage.Client().bucket(args.bucket)

    path = args.path
    out_dir = f"{path}/{args.prefix}"
    os.makedirs(out_dir, exist_ok=True)

    done = set()
    if args.skip:
        with open(args.skip) as f:
            done = {line.strip() for line in f if line.strip()}

    nb_hours_before = 12
    nb_hours_after = 24

    df_eruptions = pd.read_csv(f'{path}/eruptions.csv', index_col=0)
    # one group per eruption; Volcano Number, not Volcano Name (names are not unique)
    groups = list(df_eruptions.groupby(['Volcano Number', 'Start Date'], sort=True))

    counts = {tag: {"saved": 0, "skipped": 0, "nodata": 0, "error": 0} for tag in ("erupt", "quiet")}

    def run(rows, start_date, tag):
        first = rows.iloc[0]
        stem = window_stem(first['Volcano Number'], first['Volcano Name'], start_date, tag)
        if f"{stem}.mseed" in done or f"{stem}.nodata" in done:
            counts[tag]["skipped"] += 1
            return
        result = fetch_window(rows, start_date, nb_hours_before, nb_hours_after, out_dir, stem)
        counts[tag][result] += 1
        # empty .nodata marker: the servers answered 204 or nothing usable, so reruns skip this window.
        # No marker on "error": the next run retries it.
        if result == "nodata":
            open(f"{out_dir}/{stem}.nodata", "w").close()
        exts = {"saved": ("xml", "mseed"), "nodata": ("nodata",)}.get(result, ())
        if exts and bucket is not None:
            # .mseed last: the skip list keys on it, so a window is never skipped without its .xml
            for ext in exts:
                bucket.blob(f"{args.prefix}/{stem}.{ext}").upload_from_filename(f"{out_dir}/{stem}.{ext}")
                os.remove(f"{out_dir}/{stem}.{ext}") # the runner disk can't hold every window
            print(f"Uploaded {stem}.{'/'.join(exts)} to gs://{args.bucket}/{args.prefix}/")

    ####################################################################
    ######################## FOR ERUPTIONS #############################
    ####################################################################

    for (_, start_date), rows in groups:
        run(rows, start_date, 'erupt')

    ########################################################################
    ######################## FOR NON-ERUPTIONS #############################
    ########################################################################

    margin_before = nb_hours_before * 3600
    margin_after = nb_hours_after * 3600

    spacing = QUIET_SPACING_DAYS * 86400
    margin = QUIET_MARGIN_DAYS * 86400

    # consecutive eruptions (A, B) of the same volcano, in date order
    for ((num_A, _), rows_A), ((num_B, start_B), rows_B) in zip(groups, groups[1:]):
        if num_A != num_B: # no next eruption for this volcano
            continue

        end_A = rows_A.iloc[0]['End Date']
        if pd.isna(end_A):
            continue

        date_end_A = UTCDateTime(end_A)
        date_B = UTCDateTime(start_B)
        # middle of the quiet gap between the end of A and the start of B (the middle of A's start
        # and B's start falls inside A when A is long)
        date_mid = date_end_A + (date_B - date_end_A) / 2

        # the whole window must fit between the end of A and the start of B
        if date_mid - margin_before < date_end_A or date_mid + margin_after > date_B:
            print(f"Window too close to an eruption for {rows_A.iloc[0]['Volcano Name']}, skipped")
            continue

        # stations of A then B; A comes first, so the window is still named after A
        rows = pd.concat([rows_A, rows_B]).drop_duplicates(subset=['Client Name', 'Network', 'Station'])
        run(rows, date_mid, 'quiet')

        # extra windows on both sides of the midpoint, each far enough from both eruptions
        for date in (date_mid - spacing, date_mid + spacing):
            if date - margin_before >= date_end_A + margin and date + margin_after <= date_B - margin:
                run(rows, date, 'quiet')

    for tag, c in counts.items():
        print(f"{tag}: {c['saved']} saved, {c['skipped']} skipped, {c['nodata']} without data (.nodata), {c['error']} failed (retried next run)")
