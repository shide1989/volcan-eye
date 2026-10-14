import obspy


def cleanup_seismic_high(tr: obspy.Trace, inv: obspy.Inventory) -> obspy.Trace:
    """Broadband/short-period seismic (>= 20 Hz): keep the 1-9 Hz volcanic band (tremor, volcanic earthquakes)."""
    tr = tr.copy()
    tr.detrend("linear")  # remove offset
    tr.taper(max_percentage=0.01)  # avoid edge artefacts before filtering
    tr.remove_response(inv, output="VEL", pre_filt=(0.005, 0.01, 8, 9.5))  # counts -> m/s
    tr.filter("bandpass", freqmin=1, freqmax=9, zerophase=True)  # drop ocean microseism (0.1-1 Hz)
    return tr


def cleanup_seismic_long(tr: obspy.Trace, inv: obspy.Inventory) -> obspy.Trace:
    """Long-period seismic (1 Hz): keep very-long-period transients (< 0.02 Hz), e.g. the Jerk precursor.

    Not a Jerk input: the flat gain ignores the STS-1 roll-off and underestimates the Jerk ~35x (see ADR-002).
    """
    tr = tr.copy()
    tr.detrend("linear")  # remove offset
    tr.taper(max_percentage=0.01)  # avoid edge artefacts before filtering
    tr.remove_sensitivity(inv)  # counts -> m/s, flat gain: keeps the slow signal intact
    tr.filter("lowpass", freq=0.02, corners=4, zerophase=True)  # drop ocean microseism (> 0.05 Hz)
    return tr


def cleanup_pressure(tr: obspy.Trace, inv: obspy.Inventory) -> obspy.Trace:
    """Barometer: counts -> Pa, centred so stations at different altitudes are comparable."""
    tr = tr.copy()
    tr.remove_response(inv)  # polynomial response, counts -> Pa
    tr.detrend("linear")  # remove altitude-dependent absolute pressure
    return tr


def cleanup(tr: obspy.Trace, inv: obspy.Inventory) -> obspy.Trace:
    """Pick the pipeline from the SEED channel code: band code (1st letter), instrument code (2nd letter)."""
    band, instrument = tr.stats.channel[:2]

    if instrument == "H" and tr.stats.sampling_rate >= 20:  # BH?, HH?, EH?, SH?
        return cleanup_seismic_high(tr, inv)
    if instrument == "H" and band == "L":  # LH?
        return cleanup_seismic_long(tr, inv)
    if instrument == "D":  # ?D? pressure
        return cleanup_pressure(tr, inv)

    raise ValueError(f"No cleanup pipeline for channel {tr.id} ({tr.stats.sampling_rate} Hz)")
