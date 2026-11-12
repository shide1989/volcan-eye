"""One parquet of 10-min features per window of gs://{bucket}/{windows}/, written to gs://{bucket}/{features}/.

Run with `PYTHONPATH=notebooks`: the features come from `utils.prepare.prepare_slots`.
A window that gives no features gets a `.failed` text file holding the reason, so reruns skip it.
"""
import argparse
import tempfile
from pathlib import Path

from google.cloud import storage
from obspy import read, read_inventory

from utils.prepare import prepare_slots


def stems(bucket, prefix, ext):
    """Stems of the blobs `{prefix}/{stem}.{ext}`."""
    return {
        Path(blob.name).stem
        for blob in bucket.client.list_blobs(bucket, prefix=f"{prefix}/")
        if blob.name.endswith(f".{ext}")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="volcan_eye")
    parser.add_argument("--windows", default="windows", help="bucket prefix of the .mseed/.xml windows")
    parser.add_argument("--features", default="features/v1", help="bucket prefix of the output; never overwrite one, use a new version")
    parser.add_argument("--limit", type=int, help="process at most N windows (for tests)")
    args = parser.parse_args()

    bucket = storage.Client().bucket(args.bucket)
    windows = stems(bucket, args.windows, "mseed")
    done = stems(bucket, args.features, "parquet") | stems(bucket, args.features, "failed")
    todo = sorted(windows - done)[: args.limit]
    print(f"{len(windows)} windows, {len(windows & done)} already done, {len(todo)} to process")

    counts = {"parquet": 0, "failed": 0}
    for i, stem in enumerate(todo, 1):
        with tempfile.TemporaryDirectory() as tmp:  # deletes the local files
            try:
                for ext in ("mseed", "xml"):
                    bucket.blob(f"{args.windows}/{stem}.{ext}").download_to_filename(f"{tmp}/{stem}.{ext}")
                df = prepare_slots(read(f"{tmp}/{stem}.mseed"), read_inventory(f"{tmp}/{stem}.xml"))
                reason = None if df is not None else "prepare_slots returned None: no LHN/LHE or HZ channel"
            except Exception as e:
                reason = f"{type(e).__name__}: {e}"

            if reason is None:
                df.to_parquet(f"{tmp}/{stem}.parquet")
                bucket.blob(f"{args.features}/{stem}.parquet").upload_from_filename(f"{tmp}/{stem}.parquet")
                counts["parquet"] += 1
                print(f"[{i}/{len(todo)}] {stem}: {len(df)} rows")
            else:
                bucket.blob(f"{args.features}/{stem}.failed").upload_from_string(reason + "\n")
                counts["failed"] += 1
                print(f"[{i}/{len(todo)}] {stem}: FAILED {reason}")

    print(f"parquet: {counts['parquet']}, failed: {counts['failed']}, skipped: {len(windows & done)}")


if __name__ == "__main__":
    main()
