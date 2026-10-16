-- Volcanoes and eruptions from the GVP exports, and the seismic stations that recorded each eruption,
-- as built by scripts/volcano_db.py. Keys are GVP numbers; a station is (FDSN client, network, station).

create table public.volcanoes (
    volcano_gvp_number    integer primary key,
    name                  text not null,
    country               text not null,
    volcanic_region_group text not null,
    volcanic_region       text not null,
    volcano_landform      text not null,
    primary_volcano_type  text,
    activity_evidence     text not null,
    last_known_eruption   text not null,  -- e.g. "2025 CE", "8300 BCE", "Unknown"
    latitude              double precision not null,
    longitude             double precision not null,
    elevation_m           double precision,
    tectonic_setting      text,
    dominant_rock_type    text,
    -- from volcanodb
    volcanodb_id          integer,
    volcanodb_url         text,
    volcanic_province     text,
    eruption_count        integer,
    max_vei               smallint,
    geologic_epoch        text,
    gvp_summary           text
);

-- columns as in the GVP export: month / day 0 = unknown, years < 0 = BCE;
-- start_date / end_date are set only when the full date is known (CE)
create table public.eruptions (
    eruption_gvp_number    integer primary key,
    volcano_gvp_number     integer not null references public.volcanoes,
    eruption_category      text not null,
    area_of_activity       text,
    vei                    smallint,
    vei_modifier           text,
    start_year_modifier    text,
    start_year             integer not null,
    start_year_uncertainty integer,
    start_month            smallint,
    start_day_modifier     text,
    start_day              smallint,
    start_day_uncertainty  integer,
    evidence_method_dating text,
    end_year_modifier      text,
    end_year               integer,
    end_year_uncertainty   integer,
    end_month              smallint,
    end_day_modifier       text,
    end_day                smallint,
    end_day_uncertainty    integer,
    start_date             date,
    end_date               date
);

create index eruptions_volcano_idx on public.eruptions (volcano_gvp_number, start_date);

create table public.stations (
    client_name text not null,
    network     text not null,
    station     text not null,
    primary key (client_name, network, station)
);

create table public.eruption_stations (
    eruption_gvp_number integer not null references public.eruptions,
    client_name         text not null,
    network             text not null,
    station             text not null,
    primary key (eruption_gvp_number, client_name, network, station),
    foreign key (client_name, network, station) references public.stations
);

-- the primary key covers lookups by eruption; this one covers lookups by station
create index eruption_stations_station_idx on public.eruption_stations (client_name, network, station);

-- RLS on with no policies: not readable through the Supabase Data API; direct DB connections (postgres role) are unaffected
alter table public.volcanoes enable row level security;
alter table public.eruptions enable row level security;
alter table public.stations enable row level security;
alter table public.eruption_stations enable row level security;
