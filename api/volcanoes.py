"""Volcano data read from the DB (DATABASE_URL), shared by the /volcanoes route and the Gradio UI."""

import functools
import os

import psycopg

LIMIT = 30  # volcanoes shown: the ones with the most eruptions recorded by stations (eruption_stations)

VOLCANOES_SQL = """
with top as (
    select e.volcano_gvp_number, count(distinct e.eruption_gvp_number) as n
    from eruption_stations es join eruptions e using (eruption_gvp_number)
    group by e.volcano_gvp_number
    order by n desc, e.volcano_gvp_number
    limit %s
)
select v.volcano_gvp_number, v.name, v.country, v.latitude, v.longitude,
       v.primary_volcano_type, v.elevation_m, v.gvp_summary,
       (select coalesce(array_agg(distinct e.start_date order by e.start_date desc), '{}')
          from eruptions e
         where e.volcano_gvp_number = v.volcano_gvp_number and e.start_date is not null),
       (select coalesce(array_agg(distinct array[es.network, es.station] order by array[es.network, es.station]), '{}')
          from eruption_stations es join eruptions e using (eruption_gvp_number)
         where e.volcano_gvp_number = v.volcano_gvp_number)
from volcanoes v join top using (volcano_gvp_number)
order by v.name
"""


def connect():
    return psycopg.connect(os.environ["DATABASE_URL"])


@functools.cache  # the DB only changes when scripts/volcano_db.py reloads it: restart the app then
def list_volcanoes():
    """The displayed volcanoes, with their past eruption dates (newest first) and stations as [network, station]."""
    with connect() as conn:
        rows = conn.execute(VOLCANOES_SQL, (LIMIT,)).fetchall()
    return [
        {
            "id": id, "name": name, "country": country, "lat": lat, "lon": lon,
            "primary_volcano_type": type, "elevation_m": elev, "gvp_summary": summary,
            "eruptions": [d.isoformat() for d in eruptions],
            "stations": stations,
        }
        for id, name, country, lat, lon, type, elev, summary, eruptions, stations in rows
    ]


def has_eruption(volcano_id, day):
    """True if the volcano has an eruption starting on `day` (dates are day precision)."""
    with connect() as conn:
        sql = "select exists (select 1 from eruptions where volcano_gvp_number = %s and start_date = %s)"
        return conn.execute(sql, (volcano_id, day)).fetchone()[0]
