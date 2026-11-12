"""Checks scripts/volcano_db.py on a few volcanoes, against values read by hand from data/*.csv."""

import datetime

import pandas as pd
import pytest

from scripts import volcano_db


@pytest.fixture(scope="module")
def tables():
    return volcano_db.build_tables()


def one(df, column, value):
    rows = df[df[column] == value]
    assert len(rows) == 1
    return rows.iloc[0]


def station_counts(tables, volcano_gvp_number):
    """{eruption start date: number of stations}"""
    eruptions = tables["eruptions"].query("volcano_gvp_number == @volcano_gvp_number")
    links = tables["eruption_stations"].merge(eruptions, on="eruption_gvp_number")
    return {str(d): n for d, n in links.groupby("start_date").size().items()}


def test_all_gvp_volcanoes_and_eruptions_of_eruptions_csv(tables):
    assert len(tables["volcanoes"]) == 1214
    # only the GVP eruptions that eruptions.csv links to stations
    assert len(tables["eruptions"]) == 609
    assert tables["eruption_stations"]["eruption_gvp_number"].nunique() == 609
    assert tables["eruptions"]["start_date"].notna().all()


def test_etna(tables):
    v = one(tables["volcanoes"], "volcano_gvp_number", 211060)
    assert (v["name"], v["country"], v["volcanodb_id"]) == ("Etna", "Italy", 14)
    assert v["volcanic_province"] == "Sicily Volcanic Province"
    e = one(tables["eruptions"], "eruption_gvp_number", 13835)
    assert (e["volcano_gvp_number"], e["start_date"]) == (211060, datetime.date(2001, 12, 16))
    counts = station_counts(tables, 211060)
    assert len(counts) == 10 and set(counts.values()) == {1143}


def test_bce_eruption_has_no_date():
    e = one(volcano_db.read_gvp_eruptions(), "Eruption Number", 13500)  # Etna, 425 BCE
    assert (e["Start Year"], e["Start Month"], e["Start Day"]) == (-425, 3, 15)
    assert e["Start Date"] is None


def test_akan(tables):
    assert station_counts(tables, 285070) == {"2006-03-21": 2, "2008-11-18": 2, "2025-09-15": 2, "2026-03-07": 2}


def test_tofua_arc_is_unnamed_in_volcanodb(tables):
    v = one(tables["volcanoes"], "volcano_gvp_number", 243030)
    assert (v["name"], v["volcanodb_id"]) == ("Tofua Arc at 20.9°S", 1562)
    assert station_counts(tables, 243030) == {"2017-01-23": 10}


def test_namesake_rows_are_dropped(tables):
    # eruptions.csv: Cerro Azul (Ecuador) 2008-05-29 has 32 rows at its own coordinates + 76 at
    # Cerro Azul (Chile)'s; San Cristobal (Nicaragua) has 42 + 1 at San Cristobal (Ecuador)'s per eruption
    e = one(tables["eruptions"], "eruption_gvp_number", 11720)
    assert (e["volcano_gvp_number"], e["vei"], e["end_date"]) == (353060, 1, datetime.date(2008, 6, 17))
    assert station_counts(tables, 353060) == {"2008-05-29": 32}
    assert set(station_counts(tables, 344020).values()) == {42}


def test_volcanodb_duplicates_keep_lowest_id(tables):
    # Kaba is listed twice on volcanodb (IDs 38 and 383)
    assert one(tables["volcanoes"], "volcano_gvp_number", 261220)["volcanodb_id"] == 38


def test_every_csv_row_is_linked_or_dropped(tables):
    assert len(tables["eruption_stations"]) == len(pd.read_csv(volcano_db.ERUPTIONS_CSV)) - 100
    assert not tables["eruption_stations"].duplicated(["eruption_gvp_number"] + volcano_db.STATION_KEYS).any()


def station_set(df):
    return set(df[volcano_db.STATION_KEYS].itertuples(index=False, name=None))


def test_volcano_stations_match_eruption_stations(tables):
    vs = tables["volcano_stations"]
    assert not vs.duplicated().any()
    links = tables["eruption_stations"].merge(tables["eruptions"], on="eruption_gvp_number")
    pairs = links[["volcano_gvp_number"] + volcano_db.STATION_KEYS].itertuples(index=False, name=None)
    assert set(pairs) == set(vs.itertuples(index=False, name=None))
    # 17,178 (volcano, station) pairs in eruptions.csv, minus the namesake rows' ones
    assert len(vs) == 17101
    assert (vs["volcano_gvp_number"] == 263250).sum() == 122  # Merapi


def test_volcano_stations_skip_namesake_rows(tables):
    rows = pd.read_csv(volcano_db.ERUPTIONS_CSV).rename(columns=volcano_db.snake)
    vs = tables["volcano_stations"]
    # Cerro Azul (Ecuador) and San Cristobal (Nicaragua): only the rows at their own coordinates
    for volcano, latitude, n in [(353060, -0.92, 32), (344020, 12.702, 42)]:
        own = rows[(rows["volcano_number"] == volcano) & (rows["latitude"] == latitude)]
        stations = station_set(vs[vs["volcano_gvp_number"] == volcano])
        assert stations == station_set(own) and len(stations) == n


def probe(tables, volcano_gvp_number, start_date, client_name, network, station):
    """The eruption_stations row of one station at one eruption."""
    e = tables["eruptions"].query("volcano_gvp_number == @volcano_gvp_number and start_date == @start_date")
    links = tables["eruption_stations"].merge(e[["eruption_gvp_number"]], on="eruption_gvp_number")
    rows = links[(links["client_name"] == client_name) & (links["network"] == network) & (links["station"] == station)]
    assert len(rows) == 1
    return rows.iloc[0]


def test_every_row_has_a_probe_status(tables):
    es = tables["eruption_stations"]
    assert set(es["probe_status"]) == {"data", "nodata", "error"}
    assert (es["channels"].notna() == (es["probe_status"] == "data")).all()


def test_probe_status_counts(tables):
    # channel_survey.csv: 2,227 data / 59,949 nodata / 448 error stations, minus the 100 namesake rows'
    # (1 data: San Cristobal 2001-05-11 EARTHSCOPE YH.G01, 99 nodata)
    counts = tables["eruption_stations"]["probe_status"].value_counts().to_dict()
    assert counts == {"data": 2226, "nodata": 59850, "error": 448}


def test_fournaise_rer_channels(tables):
    p = probe(tables, 233020, datetime.date(2014, 6, 21), "EARTHSCOPE", "G", "RER")
    assert p["probe_status"] == "data"
    assert p["channels"] == ["BHE", "BHN", "BHZ", "LHE", "LHN", "LHZ", "LKI"]


def test_nodata_and_error_have_no_channels(tables):
    p = probe(tables, 221115, datetime.date(2007, 8, 12), "RESIF", "YP", "S015")  # Manda Hararo
    assert p["probe_status"] == "nodata" and p["channels"] is None
    p = probe(tables, 223020, datetime.date(2014, 6, 22), "GEOFON", "KV", "BUTK")  # Nyamulagira: 403
    assert p["probe_status"] == "error" and p["channels"] is None
