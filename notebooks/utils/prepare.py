from obspy import Stream, Trace
import pandas as pd

from utils.cleanup import cleanup
from utils.transform import transform, transform_dumb

COLUMNS_DUMB = ["EHZ_random", "LHE_random", "LHN_random"]
COLUMNS = [
    "EHZ_rms_1_3",
    "EHZ_rms_3_5",
    "EHZ_rms_5_9",
    "EHZ_kurt",
    "LHE_mean",
    "LHE_std",
    "LHE_slope",
    "LHN_mean",
    "LHN_std",
    "LHN_slope",
]


def prepare(
    st: Stream,
    inv,
    channels=("LH[NE]", "EHZ"),
    dumb=True,
) -> pd.DataFrame | None:
    """
    ### Input:
        st: Raw stream,
        inv: inventory,
        channels: Channels to filter
        - default: vertical high-rate channel only (EHZ/BHZ), merge any gaps into one trace
    ### Returns:
        - One row per 10-min window: EHZ, LHE, LHN, ehz_valid, lh_valid.
        - `None` if no usable channel."""
    lst = Stream()
    for channel in channels:
        lst += st.select(channel=channel)
        # st = lst.select(channel="LH[NE]") + st.select(channel="EHZ")
    if not lst.traces:
        return None

    # normalize sampling rates (header rounding noise breaks merge), then merge gaps into one trace per channel
    for tr in lst:
        tr.stats.sampling_rate = round(tr.stats.sampling_rate, 2)
    lst = lst.merge(method=1, fill_value=0)

    for tr in lst:
        tr.data = cleanup(tr, inv).data

    # Smart or dumb ?
    if dumb:

        df = transform_dumb(lst).reindex(columns=COLUMNS_DUMB)
    else:
        df = transform(lst).reindex(columns=COLUMNS)

    df["ehz_valid"] = df.filter(like="EHZ_").notna().any(axis=1).astype("float32")
    df["lh_valid"] = df.filter(like="LHE_").notna().any(axis=1).astype("float32")
    return df.fillna(0.0)
