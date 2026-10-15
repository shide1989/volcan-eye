import numpy as np
import obspy
import pandas as pd
from scipy.stats import kurtosis

WINDOW_S = 600  # 10-min feature step - 600 seconds
MIN_COVERAGE = 0.9  # windows with less data (trace edges) -> NaN
BANDS_HZ = ((1, 3), (3, 5), (5, 9))  # sub-bands of the 1-9 Hz volcanic band kept by cleanup_seismic_high


def features_seismic_high(tr: obspy.Trace, w: np.ndarray) -> pd.DataFrame:
    """?H? >= 20 Hz: log10 RMS velocity per sub-band (tremor / swarm level), kurtosis (spiky earthquakes vs steady tremor)."""
    feats = {}
    for fmin, fmax in BANDS_HZ:
        band = tr.copy().filter("bandpass", freqmin=fmin, freqmax=fmax, zerophase=True)
        feats[f"rms_{fmin}_{fmax}"] = np.log10(pd.Series(band.data**2).groupby(w).mean() ** 0.5)
    feats["kurt"] = pd.Series(tr.data).groupby(w).apply(kurtosis)
    return pd.DataFrame(feats)


def features_seismic_long(tr: obspy.Trace, w: np.ndarray) -> pd.DataFrame:
    """LH?: mean, std and slope (m/s per s) of the lowpassed velocity, i.e. very-long-period transients."""
    df = pd.DataFrame({"t": tr.times(), "x": tr.data})
    g = df.groupby(w)
    return pd.DataFrame({
        "mean": g["x"].mean(),
        "std": g["x"].std(),
        "slope": g.apply(lambda d: np.polyfit(d["t"], d["x"], 1)[0]),  # OLS trend within the window
    })


def features_pressure(tr: obspy.Trace, w: np.ndarray) -> pd.DataFrame:
    """Barometer: mean pressure (Pa), kept as a weather covariate."""
    return pd.DataFrame({"mean": pd.Series(tr.data).groupby(w).mean()})


def features(tr: obspy.Trace, w: np.ndarray) -> pd.DataFrame:
    """Pick the feature set from the SEED channel code, same rules as cleanup.cleanup."""
    band, instrument = tr.stats.channel[:2]

    if instrument == "H" and tr.stats.sampling_rate >= 20:  # BH?, HH?, EH?, SH?
        return features_seismic_high(tr, w)
    if instrument == "H" and band == "L":  # LH?
        return features_seismic_long(tr, w)
    if instrument == "D":  # ?D? pressure
        return features_pressure(tr, w)

    raise ValueError(f"No feature set for channel {tr.id} ({tr.stats.sampling_rate} Hz)")


def transform(st_clean: obspy.Stream) -> pd.DataFrame:
    """Cleaned traces -> one row per 10-min window on a fixed UTC grid, one column per channel feature.

    Expects one gap-free trace per channel (the output of utils.cleanup). Windows covered at less
    than MIN_COVERAGE, e.g. at trace edges, are NaN.
    """
    channels = [tr.stats.channel for tr in st_clean]
    if len(channels) != len(set(channels)):
        raise ValueError(f"Expected one trace per channel, got {channels}: merge and pick one location code first")

    # shared grid: from the first window start to the last full window, so channels with offset starts line up
    t0 = obspy.UTCDateTime(min(tr.stats.starttime for tr in st_clean).timestamp // WINDOW_S * WINDOW_S)
    n = int((max(tr.stats.endtime for tr in st_clean) - t0) // WINDOW_S)

    columns = {}
    for tr in st_clean:
        w = (((tr.stats.starttime - t0) + tr.times()) // WINDOW_S).astype(int)  # window index of each sample
        coverage = pd.Series(w).value_counts() / (tr.stats.sampling_rate * WINDOW_S)
        feats = features(tr, w).reindex(range(n))  # drops samples past the last full window
        feats[coverage.reindex(range(n)).fillna(0).to_numpy() < MIN_COVERAGE] = np.nan
        for name, col in feats.items():
            columns[f"{tr.stats.channel}_{name}"] = col.to_numpy()

    index = pd.date_range(t0.datetime, periods=n, freq=f"{WINDOW_S}s", tz="UTC", name="time")
    return pd.DataFrame(columns, index=index)
