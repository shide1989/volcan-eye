import glob
import numpy as np

from obspy import UTCDateTime
from obspy.clients.fdsn import Client

from tensorflow.keras.utils import pad_sequences, to_categorical
from sklearn.model_selection import train_test_split

from keras import Sequential, Input, Model, optimizers
from keras.layers import Dense, GRU, Dropout
from tensorflow.keras.layers import Masking

from notebooks.utils.cleanup import cleanup
from notebooks.utils.transform import transform


def init_model(input_shape: tuple) -> Model:
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Masking(mask_value=-1))
    model.add(GRU(units=20, activation="tanh", return_sequences=True))
    model.add(GRU(units=20, activation="tanh"))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    print("✅ Model initialized")

    return model


def compile_model(model: Model, learning_rate=0.0005) -> Model:
    """
    Compile the Neural Network
    """
    model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=["accuracy"])

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=2,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ) -> Model:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print("Training model...")

    model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=10,
        batch_size=batch_size,
        verbose=0
    )

    print(f"✅ Model trained on {len(X)} rows with an accuracy of X")

    return model

def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ) -> list:
    """
    Evaluate trained model performance on the dataset
    """

    print(f"Evaluating model on {len(X)} rows...")

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    accuracy = metrics["accuracy"]

    print(f"✅ Model evaluated, Accuracy: {round(accuracy, 2)} | Loss: {round(loss, 2)}")

    return metrics

def predict_new(model: Model, client_name, network, station, channel_list="*HZ"):
    now = UTCDateTime()
    start = now - 14 * 3600

    client = Client(client_name)

    st = client.get_waveforms(
        network=network,
        station=station,
        location="*",
        channel=channel_list,
        starttime=start,
        endtime=now,
    )
    inv = client.get_stations(network=network, station=station, level="response")

    st = st.select(channel="*HZ")
    st = st.merge(method=1, fill_value=0)

    if not st or st[0].stats.sampling_rate < 20:
        raise ValueError("no high-rate vertical channel")

    for tr in st:
        tr.data = cleanup(tr, inv).data

    seq = transform(st).fillna(0.0).to_numpy(dtype="float32")

    proba = float(model.predict(seq, verbose=0)[0][0])
    label = int(proba > 0.5)
    return ("eruption" if label == 1 else "quiet"), proba
