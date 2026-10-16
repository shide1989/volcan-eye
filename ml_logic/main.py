import tempfile
from pathlib import Path

import numpy as np
from google.cloud import storage
from obspy import read, read_inventory, Stream
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import pad_sequences

from ml_logic.params import BUCKET_NAME
from ml_logic.model import init_model, compile_model, train_model, evaluate_model
from ml_logic.registry import (
    load_model,
    save_model,
    save_results,
    mlflow_run,
    mlflow_transition_model,
)
from notebooks.utils.cleanup import cleanup
from notebooks.utils.transform import transform

DATA_PREFIX = "data_eruptions/"


def load_data() -> tuple[np.ndarray, np.ndarray]:
    """Download eruption .mseed/.xml blobs from GCS and return padded (X, y)."""
    bucket = storage.Client().bucket(BUCKET_NAME)
    local_dir = Path(tempfile.mkdtemp())

    for blob in bucket.list_blobs(prefix=DATA_PREFIX):
        name = Path(blob.name).name
        if name.endswith((".mseed", ".xml")):
            blob.download_to_filename(str(local_dir / name))

    X, y = [], []
    for f in sorted(local_dir.glob("*.mseed")):
        st = read(str(f))
        inv = read_inventory(str(f).replace(".mseed", ".xml"))

        # vertical high-rate channel only (EHZ/BHZ), merge any gaps into one trace
        st = st.select(channel="*HZ").merge(method=1, fill_value=0)
        st = Stream([tr for tr in st if tr.stats.sampling_rate >= 20])  # LHZ: long-period, skip

        if not st:
            continue

        for tr in st:
            tr.data = cleanup(tr, inv).data

        X.append(transform(st).fillna(0.0).to_numpy(dtype="float32"))  # edge windows -> 0
        y.append(1)  # all current files are eruptions

    if not X:
        return np.empty((0,)), np.empty((0,))

    X = pad_sequences(X, padding="post", value=-1.0, dtype="float32")
    return X, np.array(y, dtype="float32")


@mlflow_run
def train(
        learning_rate: float = 0.0005,
        batch_size: int = 256,
        patience: int = 2,
    ) -> float:

    print("Fetching eruption data from cloud storage...")
    X, y = load_data()
    print(len(X), "sequences/files |", int(y.sum()), "eruptions,", len(y) - int(y.sum()), "quiet")

    if len(X) == 0:
        print("❌ No data to train on")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = load_model()

    if model is None:
        model = init_model(input_shape=X_train.shape[1:])

    model = compile_model(model, learning_rate=learning_rate)
    model = train_model(
        model, X_train, y_train,
        batch_size=batch_size,
        patience=patience,
        validation_data=(X_test, y_test),
    )

    accuracy = max(model.history.history['accuracy'])

    save_results(
        params={'context': "train", 'row_count': len(X_train)},
        metrics={'accuracy': accuracy},
    )
    save_model(model=model)
    mlflow_transition_model(current_stage="None", new_stage="Staging")

    print("✅ train() done \n")

    return accuracy


@mlflow_run
def evaluate(
        stage: str = "Production",
    ) -> float:

    model = load_model(stage=stage)

    if model is None:
        print("❌ No model to evaluate")
        return None

    print("Fetching eruption data from cloud storage...")
    X, y = load_data()

    if len(X) == 0:
        print("❌ No data to evaluate on")
        return None

    metrics_dict = evaluate_model(model=model, X=X, y=y)

    save_results(
        params={'context': "evaluate", 'row_count': len(X)},
        metrics=metrics_dict,
    )

    print("✅ evaluate() done \n")

    return metrics_dict["accuracy"]


if __name__ == '__main__':
    train()
    evaluate()
