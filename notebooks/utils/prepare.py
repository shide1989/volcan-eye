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


HZ_CODES = ("EHZ", "BHZ", "HHZ")  # high-rate vertical slot, in download_windows.SLOTS order
COLUMNS_SLOTS = [c.replace("EHZ_", "HZ_") for c in COLUMNS]


def prepare_slots(st: Stream, inv) -> pd.DataFrame | None:
    """Like prepare(dumb=False), with EHZ, BHZ or HHZ as one `HZ` input (ADR-005).

    Keeps LHN/LHE plus the first code of HZ_CODES present at >= 20 Hz. The traces keep their
    real SEED code (cleanup and transform pick their pipeline from it); only the columns are renamed.
    ### Returns:
        - One row per 10-min window: COLUMNS_SLOTS, hz_valid, lh_valid.
        - `None` if no usable channel."""
    lst = st.select(channel="LH[NE]")
    for code in HZ_CODES:
        hz = Stream([tr for tr in st.select(channel=code) if tr.stats.sampling_rate >= 20])
        if hz:
            lst += hz
            break
    if not lst.traces:
        return None

    # same steps as prepare()
    for tr in lst:
        tr.stats.sampling_rate = round(tr.stats.sampling_rate, 2)
    lst = lst.merge(method=1, fill_value=0)

    for tr in lst:
        tr.data = cleanup(tr, inv).data

    df = transform(lst)
    df.columns = [f"HZ_{c[4:]}" if c[:4] in {f"{code}_" for code in HZ_CODES} else c for c in df.columns]
    df = df.reindex(columns=COLUMNS_SLOTS)

    df["hz_valid"] = df.filter(like="HZ_").notna().any(axis=1).astype("float32")
    df["lh_valid"] = df.filter(like="LHE_").notna().any(axis=1).astype("float32")
    return df.fillna(0.0)
