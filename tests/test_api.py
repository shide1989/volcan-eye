import datetime
import os

import pytest
from fastapi.testclient import TestClient

from api import predict
from api.app import app
from api.ui import level

client = TestClient(app)


def test_addition():
    assert 1 + 1 == 2


def test_seismic_predict_is_unchanged():
    assert client.get("/seismic/predict").json() == {"prediction": "eruption de ouf"}


def test_replay_is_1_on_an_eruption_day(monkeypatch):
    monkeypatch.setattr(predict, "has_eruption", lambda volcano_id, day: day == datetime.date(2023, 7, 2))
    assert predict.predict_proba(233020, datetime.date(2023, 7, 2)) == 1.0
    assert predict.predict_proba(233020, datetime.date(2023, 7, 3)) == 0.0


def test_predict_rejects_today_and_has_no_model_yet():
    today = datetime.datetime.now(datetime.timezone.utc).date()
    with pytest.raises(ValueError):
        predict.predict_proba(233020, today)
    with pytest.raises(NotImplementedError):
        predict.predict_proba(233020)


def test_levels():
    assert [level(p)[0] for p in [0, 0.29, 0.3, 0.7, 0.71, 1]] == \
        ["calm", "calm", "restless", "restless", "eruption", "eruption"]


def test_ui_is_served():
    assert client.get("/ui/").status_code == 200


@pytest.mark.skipif("DATABASE_URL" not in os.environ, reason="needs the DB")
def test_volcanoes_has_piton_de_la_fournaise():
    volcanoes = client.get("/volcanoes").json()
    assert len(volcanoes) == 30
    fournaise = next(v for v in volcanoes if v["id"] == 233020)
    assert fournaise["country"] == "France"
    assert "2023-07-02" in fournaise["eruptions"]
    assert ["G", "RER"] in fournaise["stations"]
