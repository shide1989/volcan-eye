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


def test_all_gvp_volcanoes_and_eruptions(tables):
    assert len(tables["volcanoes"]) == 1214
    # 9918 GVP eruptions, minus 7 of volcanoes missing from the (newer) volcano list
    assert len(tables["eruptions"]) == 9918 - 7


def test_etna(tables):
    v = one(tables["volcanoes"], "volcano_gvp_number", 211060)
    assert (v["name"], v["country"], v["volcanodb_id"]) == ("Etna", "Italy", 14)
    assert v["volcanic_province"] == "Sicily Volcanic Province"
    e = one(tables["eruptions"], "eruption_gvp_number", 13835)
    assert (e["volcano_gvp_number"], e["start_date"]) == (211060, datetime.date(2001, 12, 16))
    counts = station_counts(tables, 211060)
    assert len(counts) == 10 and set(counts.values()) == {1143}


def test_bce_eruption_has_no_date(tables):
    e = one(tables["eruptions"], "eruption_gvp_number", 13500)  # Etna, 425 BCE
    assert (e["start_year"], e["start_month"], e["start_day"]) == (-425, 3, 15)
    assert e["start_date"] is None


def test_akan(tables):
    assert station_counts(tables, 285070) == {"2006-03-21": 2, "2008-11-18": 2, "2025-09-15": 2, "2026-03-07": 2}


def test_tofua_arc_is_unnamed_in_volcanodb(tables):
    v = one(tables["volcanoes"], "volcano_gvp_number", 243030)
    assert (v["name"], v["volcanodb_id"]) == ("Tofua Arc at 20.9°S", 1562)
    assert station_counts(tables, 243030) == {"2017-01-23": 10}


def test_namesake_rows_are_dropped(tables):
    # eruptions_new.csv: Cerro Azul (Ecuador) 2008-05-29 has 32 rows at its own coordinates + 76 at
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
    assert not tables["eruption_stations"].duplicated().any()
