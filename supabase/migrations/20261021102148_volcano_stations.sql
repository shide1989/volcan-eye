-- The stations of each volcano, from eruptions.csv as built by scripts/volcano_db.py: a station can belong
-- to several volcanoes, and reaches them without going through eruption_stations.

create table public.volcano_stations (
    volcano_gvp_number integer not null references public.volcanoes,
    client_name        text not null,
    network            text not null,
    station            text not null,
    primary key (volcano_gvp_number, client_name, network, station),
    foreign key (client_name, network, station) references public.stations
);

-- the primary key covers lookups by volcano; this one covers lookups by station
create index volcano_stations_station_idx on public.volcano_stations (client_name, network, station);

-- RLS on with no policies, as for the other tables
alter table public.volcano_stations enable row level security;
