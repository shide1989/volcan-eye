import keras
from keras import Input, Sequential, regularizers
from keras.layers import GRU, Dense, Masking

from utils.normalize import MASK_VALUE


def init_model(n_steps, n_features):
    """The GRU of notebooks/a_scaled_rnn.ipynb."""
    keras.utils.set_random_seed(42)  # same initial weights for every run
    model = Sequential()
    model.add(Input(shape=(n_steps, n_features)))
    model.add(Masking(mask_value=MASK_VALUE))
    model.add(GRU(units=32, activation="tanh", return_sequences=True))
    model.add(GRU(units=16, activation="tanh"))
    model.add(Dense(64, activation="relu", kernel_regularizer=regularizers.L1(0.01)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["auc"])
    return model
