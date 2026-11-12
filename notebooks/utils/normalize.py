import numpy as np
from keras.layers import Normalization
from keras.utils import pad_sequences

MASK_VALUE = -999.0  # padding value, never reached by normalized data


def fit_normalizer(X: list[np.ndarray]) -> Normalization:
    """One mean / std per feature, fitted on the real time steps of the TRAIN sequences only (no leakage)."""
    normalizer = Normalization(axis=-1)
    normalizer.adapt(np.concatenate(X, axis=0))  # (total_steps, n_features)
    return normalizer


def normalize_and_pad(
    X: list[np.ndarray], normalizer: Normalization, maxlen: int
) -> np.ndarray:
    """Normalize each sequence, then pad/truncate at the end to `maxlen`.
    Padding comes last so MASK_VALUE is neither in the statistics nor normalized.
    Returns (n_sequences, maxlen, n_features)."""
    X = [normalizer(x).numpy() for x in X]
    return pad_sequences(
        X,
        maxlen=maxlen,
        dtype="float32",
        padding="post",
        truncating="post",
        value=MASK_VALUE,
    )
