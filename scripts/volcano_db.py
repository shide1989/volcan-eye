"""Read the volcano / eruption / station CSVs in data/ and link them into DB-ready tables:
volcanoes, eruptions (only the ones in eruptions.csv), stations, eruption_stations, volcano_stations.

Sources:
- GVP_Volcano_List_Holocene_*.csv: all Holocene volcanoes (GVP "Volcano Number").
- GVP_Eruption_List_Holocene_*.csv: all their eruptions (GVP "Eruption Number").
- volcanodb_volcanoes.csv + volcanodb_volcanoes_details.csv: scraped volcanodb pages, only for the fields
  GVP lists don't have (province, eruption count, max VEI, epoch, summary, volcanodb ID/URL).
- eruptions.csv: one row per (eruption start date, volcano, FDSN client, network, station).
- channel_survey.csv: the 10-min get_waveforms probe of each eruptions.csv row at the eruption start date
  (one row per channel returned, or one nodata / error row); `gcloud storage cp gs://volcan_eye/channel_survey.csv data/`.

Everything is joined on GVP numbers; names are not unique ("San Cristobal" x2).

Run `python scripts/volcano_db.py --load` to replace the DB tables (supabase/migrations) with this data;
the target DB is DATABASE_URL.
"""

import datetime
import os
import re
import sys
from pathlib import Path

import pandas as pd
import psycopg

DATA = Path(__file__).resolve().parent.parent / "data"

GVP_VOLCANOES_CSV = DATA / "GVP_Volcano_List_Holocene_202610130935.csv"
GVP_ERUPTIONS_CSV = DATA / "GVP_Eruption_List_Holocene_20260424.csv"
VOLCANODB_VOLCANOES_CSV = DATA / "volcanodb_volcanoes.csv"
VOLCANODB_DETAILS_CSV = DATA / "volcanodb_volcanoes_details.csv"
ERUPTIONS_CSV = DATA / "eruptions.csv"
CHANNEL_SURVEY_CSV = DATA / "channel_survey.csv"

STATION_KEYS = ["client_name", "network", "station"]  # the same network.station can be served by 2 clients


def snake(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def to_date(year, month, day):
    """GVP date parts -> date; None when partial (month or day 0), missing or BCE."""
    try:
        return datetime.date(int(year), int(month), int(day))
    except (TypeError, ValueError):
        return None


def read_gvp_volcanoes():
    return pd.read_csv(GVP_VOLCANOES_CSV, index_col=0)


def read_gvp_eruptions():
    # row 0 is a title banner; empty values are written "NULL"
    df = pd.read_csv(GVP_ERUPTIONS_CSV, skiprows=1, encoding="utf-8-sig", na_values=["NULL"], keep_default_na=False)
    for when in ["Start", "End"]:
        parts = df[[f"{when} Year", f"{when} Month", f"{when} Day"]].itertuples(index=False)
        df[f"{when} Date"] = [to_date(*p) for p in parts]
    return df


def read_volcanodb():
    """volcanodb volcano list + details, one row per GVP volcano number."""
    volcanoes = pd.read_csv(VOLCANODB_VOLCANOES_CSV)
    details = pd.read_csv(VOLCANODB_DETAILS_CSV)
    details = details.drop(columns=volcanoes.columns.drop("Volcano ID"))
    df = volcanoes.merge(details, on="Volcano ID", validate="one_to_one")
    # rows with no GVP number are near-empty; duplicates per GVP number are identical (ADR-003)
    df = df.dropna(subset=["GVP Volcano Number"]).sort_values("Volcano ID")
    df = df.drop_duplicates("GVP Volcano Number")
    df["GVP Volcano Number"] = df["GVP Volcano Number"].astype(int)
    return df


def read_eruption_stations():
    df = pd.read_csv(ERUPTIONS_CSV, index_col=0)
    df["Start Date"] = pd.to_datetime(df["Start Date"]).dt.date
    return df


def read_channel_survey():
    """One row per (volcano, eruption start date, FDSN client, network, station): probe_status (data / nodata /
    error) and channels, the sorted channel codes that returned data (None unless probe_status is data)."""
    if not CHANNEL_SURVEY_CSV.exists():
        raise FileNotFoundError(f"{CHANNEL_SURVEY_CSV} is missing: gcloud storage cp gs://volcan_eye/channel_survey.csv data/")
    df = pd.read_csv(CHANNEL_SURVEY_CSV, dtype={"network": str, "station": str, "location": str, "channel": str,
                                                 "detail": str})
    df["start_date"] = pd.to_datetime(df["start_date"]).dt.date
    survey = df.groupby(["volcano_number", "start_date", "client", "network", "station"]).agg(
        probe_status=("status", "unique"), channels=("channel", lambda c: sorted(c.dropna().unique()))
    )
    mixed = survey["probe_status"].map(len) > 1
    assert not mixed.any(), f"{mixed.sum()} {CHANNEL_SURVEY_CSV.name} stations mix probe statuses"
    survey["probe_status"] = survey["probe_status"].str[0]
    survey["channels"] = survey["channels"].where(survey["probe_status"] == "data", None)
    return survey.reset_index().rename(columns={"client": "client_name"})


def build_tables():
    """Link the CSVs into the tables we will store: volcanoes, eruptions, stations, eruption_stations,
    volcano_stations."""
    gvp_volcanoes = read_gvp_volcanoes()
    gvp_eruptions = read_gvp_eruptions()
    rows = read_eruption_stations()

    volcanodb = read_volcanodb()[
        ["GVP Volcano Number", "Volcano ID", "URL", "Volcanic Province", "Eruption Count", "Max VEI",
         "Geologic Epoch", "GVP Summary"]
    ].rename(columns={"GVP Volcano Number": "Volcano Number", "Volcano ID": "volcanodb_id", "URL": "volcanodb_url"})
    volcanoes = gvp_volcanoes.merge(volcanodb, on="Volcano Number", how="left", validate="one_to_one")
    volcanoes = volcanoes.rename(columns={"Volcano Number": "volcano_gvp_number", "Volcano Name": "name"})
    volcanoes = volcanoes.rename(columns=snake).astype(
        {"volcanodb_id": "Int64", "eruption_count": "Int64", "max_vei": "Int64"}
    )

    # the two GVP exports are from different dates: a few eruptions belong to volcanoes no longer in the list
    orphans = ~gvp_eruptions["Volcano Number"].isin(gvp_volcanoes["Volcano Number"])
    if orphans.any():
        print(f"skipped {orphans.sum()} GVP eruptions of volcanoes not in the volcano list: "
              f"{sorted(gvp_eruptions.loc[orphans, 'Volcano Name'].unique())}")
    eruptions = gvp_eruptions[~orphans].drop(columns="Volcano Name")
    eruptions = eruptions.astype({c: "Int64" for c in eruptions.select_dtypes("float").columns})
    eruptions = eruptions.rename(columns={"Volcano Number": "volcano_gvp_number", "Eruption Number": "eruption_gvp_number"})
    eruptions = eruptions.rename(columns=snake)

    # eruptions.csv was merged on volcano names: rows whose coordinates are not their volcano's are a
    # namesake's (Cerro Azul Chile vs Ecuador, San Cristobal Ecuador vs Nicaragua) and carry its stations
    rows = rows.merge(gvp_volcanoes[["Volcano Number", "Latitude", "Longitude"]], on="Volcano Number",
                      suffixes=("", " GVP"), validate="many_to_one")
    namesake = (rows["Latitude"] != rows["Latitude GVP"]) | (rows["Longitude"] != rows["Longitude GVP"])
    print(f"dropped {namesake.sum()} {ERUPTIONS_CSV.name} rows with a namesake volcano's coordinates: "
          f"{sorted(rows.loc[namesake, 'Volcano Name'].unique())}")
    rows = rows[~namesake].merge(
        gvp_eruptions.dropna(subset=["Start Date"])[["Volcano Number", "Start Date", "Eruption Number"]],
        on=["Volcano Number", "Start Date"], how="left", validate="many_to_one",
    )
    unlinked = rows["Eruption Number"].isna()
    if unlinked.any():
        raise ValueError(f"{unlinked.sum()} {ERUPTIONS_CSV.name} rows match no GVP eruption")

    rows = rows.rename(columns={"Eruption Number": "eruption_gvp_number"}).rename(columns=snake)
    rows["eruption_gvp_number"] = rows["eruption_gvp_number"].astype(int)
    # the survey probed every eruptions.csv row
    rows = rows.merge(read_channel_survey(), on=["volcano_number", "start_date"] + STATION_KEYS, how="left",
                      validate="many_to_one")
    unprobed = rows["probe_status"].isna()
    if unprobed.any():
        raise ValueError(f"{unprobed.sum()} {ERUPTIONS_CSV.name} rows have no {CHANNEL_SURVEY_CSV.name} row")
    stations = rows[STATION_KEYS].drop_duplicates()
    eruption_stations = rows[["eruption_gvp_number"] + STATION_KEYS + ["probe_status", "channels"]]
    volcano_stations = rows[["volcano_number"] + STATION_KEYS].drop_duplicates()
    volcano_stations = volcano_stations.rename(columns={"volcano_number": "volcano_gvp_number"})
    # keep only the eruptions of eruptions.csv: the GVP ones without stations are not used
    eruptions = eruptions[eruptions["eruption_gvp_number"].isin(eruption_stations["eruption_gvp_number"])]

    return {
        "volcanoes": volcanoes.reset_index(drop=True),
        "eruptions": eruptions.reset_index(drop=True),
        "stations": stations.reset_index(drop=True),
        "eruption_stations": eruption_stations.reset_index(drop=True),
        "volcano_stations": volcano_stations.reset_index(drop=True),
    }


def load(tables, database_url):
    """Replace the content of the DB tables with `tables`, in one transaction."""
    order = ["volcanoes", "eruptions", "stations", "volcano_stations", "eruption_stations"]  # parents before children
    with psycopg.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute(f"truncate {', '.join(order)}")
        for name in order:
            df = tables[name]
            rows = df.astype(object).where(df.notna(), None).itertuples(index=False)
            with cur.copy(f"copy {name} ({', '.join(df.columns)}) from stdin") as copy:
                for row in rows:
                    copy.write_row(row)
            print(f"{name}: {len(df)} rows loaded")


if __name__ == "__main__":
    tables = build_tables()
    if "--load" in sys.argv:
        load(tables, os.environ["DATABASE_URL"])
    else:
        for name, df in tables.items():
            print(f"\n== {name}: {len(df)} rows")
            print(df.head(3).to_string())
