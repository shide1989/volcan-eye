-- The channel survey (data/channel_survey.csv) on each eruption station, as built by scripts/volcano_db.py:
-- the result of a 10-min get_waveforms probe at the eruption start date. Kept per eruption, not per station,
-- because a station's channels change over time.

alter table public.eruption_stations
    add column probe_status text check (probe_status in ('data', 'nodata', 'error')),
    add column channels     text[];  -- distinct channel codes that returned data, sorted; null unless probe_status = 'data'
