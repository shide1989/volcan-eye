"""Cross-validate and train the GRU on the parquet features of gs://{bucket}/{features}/.

Run with `PYTHONPATH=notebooks`. Selection caps follow ADR-004; folds are grouped by volcano.
Outputs go to gs://{bucket}/models/{run_id}/: model.keras, normalizer.npz, config.json, windows.csv, metrics.json.
"""
import argparse
import io
import json
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
from google.cloud import storage
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from utils.model import init_model
from utils.normalize import fit_normalizer, normalize_and_pad

FOURNAISE = 233020
EXCLUDE = {"344020_San-Cristobal_2011-01-14_quiet"}  # overlaps a GVP eruption missing from eruptions.csv
LABELS = {"erupt": 1, "quiet": 0}


def load(bucket, prefix):
    """Every `{prefix}/*.parquet` -> (windows table, {stem: (n_steps, n_features) array}, column names)."""
    rows, X, columns = [], {}, None
    for blob in bucket.client.list_blobs(bucket, prefix=f"{prefix}/"):
        if not blob.name.endswith(".parquet"):
            continue
        stem = blob.name.rsplit("/", 1)[-1].removesuffix(".parquet")
        volcano_number, volcano_name, date, tag = stem.split("_")  # window_stem strips "_" from names
        df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
        columns = columns or list(df.columns)
        assert list(df.columns) == columns, f"{stem}: columns {list(df.columns)} != {columns}"
        X[stem] = df.to_numpy(dtype="float32")
        rows.append({"stem": stem, "volcano": int(volcano_number), "date": date, "label": LABELS[tag]})
    return pd.DataFrame(rows), X, columns


def select(windows, max_erupt, quiet_per_erupt, uncapped):
    """ADR-004 caps per volcano; 0 disables a cap. Returns the selected stems."""
    selected = []
    for volcano, g in windows[~windows.stem.isin(EXCLUDE)].groupby("volcano"):
        erupt, quiet = g[g.label == 1], g[g.label == 0]
        if max_erupt and volcano not in uncapped:
            erupt = erupt.sample(min(len(erupt), max_erupt), random_state=42)
        if quiet_per_erupt:
            quiet = quiet.sample(min(len(quiet), quiet_per_erupt * len(erupt)), random_state=42)
        if erupt.empty:  # a volcano with no erupt window keeps no quiet window
            continue
        selected += erupt.stem.tolist() + quiet.stem.tolist()
    return set(selected)


def fit(X_train, y_train, X_val=None, y_val=None, epochs=500):
    """Normalizer fitted on X_train only, then the GRU. With validation data, early stopping on val_loss."""
    normalizer = fit_normalizer(X_train)
    maxlen = max(len(x) for x in X_train)
    model = init_model(maxlen, X_train[0].shape[1])
    kwargs = {}
    if X_val is not None:
        kwargs = {
            "validation_data": (normalize_and_pad(X_val, normalizer, maxlen), y_val),
            "callbacks": [EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)],
        }
    history = model.fit(normalize_and_pad(X_train, normalizer, maxlen), y_train, batch_size=8, epochs=epochs, verbose=2, **kwargs)
    return model, normalizer, maxlen, history


def git_sha():
    if "GITHUB_SHA" in os.environ:
        return os.environ["GITHUB_SHA"]
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", default="volcan_eye")
    parser.add_argument("--features", default="features/v1", help="bucket prefix of the parquet files")
    parser.add_argument("--run-id", required=True, help="output folder: gs://{bucket}/models/{run_id}/")
    parser.add_argument("--max-erupt", type=int, default=10, help="erupt windows kept per volcano (0: no cap)")
    parser.add_argument("--quiet-per-erupt", type=int, default=2, help="quiet windows kept per kept erupt window (0: no cap)")
    parser.add_argument("--uncapped", type=int, nargs="*", default=[FOURNAISE], help="volcano numbers that keep all their erupt windows")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    bucket = storage.Client().bucket(args.bucket)
    windows, X, columns = load(bucket, args.features)
    windows["selected"] = windows.stem.isin(select(windows, args.max_erupt, args.quiet_per_erupt, args.uncapped)).astype(int)
    windows["fold"] = pd.NA
    data = windows[windows.selected == 1].reset_index(drop=True)
    print(f"{len(windows)} windows loaded, {len(data)} selected: {(data.label == 1).sum()} erupt, {(data.label == 0).sum()} quiet, "
          f"{data.volcano.nunique()} volcanoes, Fournaise {(data.volcano == FOURNAISE).mean():.0%}")

    X_all, y_all, groups = [X[s] for s in data.stem], data.label.to_numpy(), data.volcano.to_numpy()

    folds = []
    for k, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=args.folds).split(X_all, y_all, groups)):
        inner, val = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(train_idx, groups=groups[train_idx]))
        inner, val = train_idx[inner], train_idx[val]
        model, normalizer, maxlen, history = fit(
            [X_all[i] for i in inner], y_all[inner], [X_all[i] for i in val], y_all[val], args.epochs
        )
        y_test = y_all[test_idx]
        proba = model.predict(normalize_and_pad([X_all[i] for i in test_idx], normalizer, maxlen), verbose=0).ravel()
        test = data.iloc[test_idx]
        folds.append({
            "fold": k,
            "auc": roc_auc_score(y_test, proba) if len(set(y_test)) == 2 else None,
            "accuracy": accuracy_score(y_test, proba >= 0.5),
            "n_erupt": int((test.label == 1).sum()),
            "n_quiet": int((test.label == 0).sum()),
            "volcanoes_erupt": int(test[test.label == 1].volcano.nunique()),
            "volcanoes_quiet": int(test[test.label == 0].volcano.nunique()),
            "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
            "fournaise_in_test": bool((test.volcano == FOURNAISE).any()),
        })
        windows.loc[windows.stem.isin(test.stem), "fold"] = k
        print(f"fold {k}: {folds[-1]}")

    aucs = [f["auc"] for f in folds if f["auc"] is not None]
    metrics = {
        "folds": folds,
        "auc_mean": float(np.mean(aucs)) if aucs else None,
        "auc_std": float(np.std(aucs)) if aucs else None,
    }
    final_epochs = int(np.median([f["best_epoch"] for f in folds]))
    print(f"AUC {metrics['auc_mean']} ± {metrics['auc_std']}; final model: {final_epochs} epochs on all {len(data)} windows")

    model, normalizer, maxlen, _ = fit(X_all, y_all, epochs=final_epochs)
    config = {
        **vars(args),
        "final_epochs": final_epochs,
        "maxlen": maxlen,
        "columns": columns,
        "git_sha": git_sha(),
    }

    out = f"models/{args.run_id}"
    with tempfile.TemporaryDirectory() as tmp:
        model.save(f"{tmp}/model.keras")
        np.savez(f"{tmp}/normalizer.npz", mean=np.asarray(normalizer.mean), variance=np.asarray(normalizer.variance))
        with open(f"{tmp}/config.json", "w") as f:
            json.dump(config, f, indent=2)
        windows[["stem", "volcano", "label", "fold", "selected"]].to_csv(f"{tmp}/windows.csv", index=False)
        with open(f"{tmp}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        for name in os.listdir(tmp):
            bucket.blob(f"{out}/{name}").upload_from_filename(f"{tmp}/{name}")
    print(f"Uploaded to gs://{args.bucket}/{out}/")

    if "GITHUB_STEP_SUMMARY" in os.environ:
        lines = [
            f"## GRU, features `{args.features}`, run `{args.run_id}`",
            f"{len(data)} windows ({(data.label == 1).sum()} erupt / {(data.label == 0).sum()} quiet), "
            f"{data.volcano.nunique()} volcanoes. Mean test AUC: **{metrics['auc_mean']:.3f}** ± {metrics['auc_std']:.3f}"
            if aucs else f"{data.volcano.nunique()} volcanoes. No fold has both labels: no AUC",
            "",
            "| fold | AUC | accuracy | erupt | quiet | best epoch | Fournaise in test |",
            "|---|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {f['fold']} | {f['auc'] if f['auc'] is None else round(f['auc'], 3)} | {f['accuracy']:.3f} | {f['n_erupt']} | "
            f"{f['n_quiet']} | {f['best_epoch']} | {f['fournaise_in_test']} |"
            for f in folds
        ]
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
