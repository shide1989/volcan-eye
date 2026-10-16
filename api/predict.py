"""Probability of an eruption at T+8 h for one volcano."""

import datetime

from api.volcanoes import has_eruption


def predict_proba(volcano_id: int, date: datetime.date | None = None,
                  network: str | None = None, station: str | None = None) -> float:
    """date=None: now (UTC) -> T+8 h. A past date: replay, 1.0 if an eruption started that day, else 0.0."""
    if date is None:
        # TODO: plug the baseline model in here (network / station select its input signal)
        raise NotImplementedError("no prediction model yet: only replays of past dates work")
    if date >= datetime.datetime.now(datetime.timezone.utc).date():
        raise ValueError(f"{date} is not in the past: use date=None to predict from now")
    return 1.0 if has_eruption(volcano_id, date) else 0.0
