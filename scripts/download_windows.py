import argparse
import os
import pandas as pd
import re
import sys
import tempfile
import unicodedata
from pathlib import Path

from obspy import UTCDateTime, Stream, Inventory
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

sys.path.insert(0, str(ROOT / "notebooks")) # get_seismo_data.py lives with the notebooks
from get_seismo_data import get_seismo_data

MAX_TRIES = 10 # max stations called per window

# channel slots kept per window: slot -> channel codes that can fill it, in order of preference
SLOTS = {
    "HZ": ["EHZ", "BHZ", "HHZ"], # high-rate vertical (1-9 Hz volcanic band), must be >= 20 Hz
    "LHN": ["LHN"],              # very-long-period horizontals (Jerk transient)
    "LHE": ["LHE"],
}
CHANNEL_LIST = ",".join(code for codes in SLOTS.values() for code in codes)

############# One window = the channels of several stations, combined into one mseed + one StationXML ################

def window_stem(volcano_number, volcano_name, start_date, tag):
    # same cleaning as in get_seismo_data
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


def operating_rows(rows, start_date, t_before, t_after):
    """rows whose station has a CHANNEL_LIST channel operating during the window, in CSV order."""
    d = UTCDateTime(start_date)
    start, end = d - t_before*3600, d + t_after*3600
    ok = set()
    for client_name, g in rows.groupby("Client Name"):
        pairs = g[["Network", "Station"]].drop_duplicates().values.tolist()
        client = Client(client_name)
        # GET with comma lists: RESIF rejects bulk POSTs of more than a few lines.
        # The net x sta cross product can return extra stations; the mask below drops them.
        for i in range(0, len(pairs), 200):
            chunk = pairs[i:i + 200]
            try:
                inv = client.get_stations(network=",".join(sorted({n for n, _ in chunk})),
                                          station=",".join(sorted({s for _, s in chunk})),
                                          channel=CHANNEL_LIST, starttime=start, endtime=end, level="station")
            except FDSNNoDataException:
                continue
            except Exception as exc:
                print(f"get_stations failed for {client_name}: {exc}")
                continue
            ok |= {(client_name, net.code, sta.code) for net in inv for sta in net}
    mask = [tuple(k) in ok for k in rows[["Client Name", "Network", "Station"]].values]
    return rows[mask]


def fetch_window(rows, start_date, tag, tmp_path, t_before, t_after, out_dir, stem):
    combined_st = Stream()
    combined_inv = Inventory(networks=[])
    filled = {} # slot -> trace id

    for _, row in operating_rows(rows, start_date, t_before, t_after).head(MAX_TRIES).iterrows():
        if len(filled) == len(SLOTS): # every slot filled: stop calling stations
            break

        st, inv = get_seismo_data(start_date = start_date,
                        t_before = t_before,
                        t_after = t_after,
                        client_name = row['Client Name'],
                        network = row['Network'],
                        station = row['Station'],
                        channel_list = CHANNEL_LIST,
                        path = tmp_path,
                        tag = tag,
                        volcano_name = row['Volcano Name']
        )
        if st is None or inv is None:
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
        return False

    print(f"Saving {stem}: {filled}")
    combined_st.write(f"{out_dir}/{stem}.mseed", format="MSEED")
    combined_inv.write(f"{out_dir}/{stem}.xml", format="STATIONXML")
    return True

#######################################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download one combined erupt / quiet window per eruption.")
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

    counts = {tag: {"saved": 0, "skipped": 0, "no data": 0} for tag in ("erupt", "quiet")}

    def run(rows, start_date, tag, tmp_path):
        first = rows.iloc[0]
        stem = window_stem(first['Volcano Number'], first['Volcano Name'], start_date, tag)
        if f"{stem}.mseed" in done:
            counts[tag]["skipped"] += 1
            return
        saved = fetch_window(rows, start_date, tag, tmp_path, nb_hours_before, nb_hours_after, out_dir, stem)
        counts[tag]["saved" if saved else "no data"] += 1
        if saved and bucket is not None:
            # .mseed last: the skip list keys on it, so a window is never skipped without its .xml
            for ext in ("xml", "mseed"):
                bucket.blob(f"{args.prefix}/{stem}.{ext}").upload_from_filename(f"{out_dir}/{stem}.{ext}")
                os.remove(f"{out_dir}/{stem}.{ext}") # the runner disk can't hold every window
            print(f"Uploaded {stem} to gs://{args.bucket}/{args.prefix}/")

    with tempfile.TemporaryDirectory() as tmp_path:
        # get_seismo_data writes its per-station files here; they are thrown away
        os.makedirs(f"{tmp_path}/data_eruptions_and_quiet")

        ####################################################################
        ######################## FOR ERUPTIONS #############################
        ####################################################################

        for (_, start_date), rows in groups:
            run(rows, start_date, 'erupt', tmp_path)

        ########################################################################
        ######################## FOR NON-ERUPTIONS #############################
        ########################################################################

        margin_before = nb_hours_before * 3600
        margin_after = nb_hours_after * 3600

        # consecutive eruptions (A, B) of the same volcano, in date order
        for ((num_A, start_A), rows_A), ((num_B, start_B), _) in zip(groups, groups[1:]):
            if num_A != num_B: # no next eruption for this volcano
                continue

            end_A = rows_A.iloc[0]['End Date']
            if pd.isna(end_A):
                continue

            date_A = UTCDateTime(start_A)
            date_end_A = UTCDateTime(end_A)
            date_B = UTCDateTime(start_B)
            date_mid = date_A + (date_B - date_A) / 2

            # the whole window must fit between the end of A and the start of B
            if date_mid - margin_before < date_end_A or date_mid + margin_after > date_B:
                print(f"Window too close to an eruption for {rows_A.iloc[0]['Volcano Name']}, skipped")
                continue

            run(rows_A, date_mid, 'quiet', tmp_path)

    for tag, c in counts.items():
        print(f"{tag}: {c['saved']} saved, {c['skipped']} skipped, {c['no data']} without valid data")
