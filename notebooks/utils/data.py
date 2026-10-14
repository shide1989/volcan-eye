import os

import obspy
from obspy.clients.fdsn import Client
from obspy import UTCDateTime, read, read_inventory

client = Client("IRIS", debug=True)  # "RESIF" ou "IRIS" ou "EPOSFR" ou "EARTHSCOPE"


def get_help():
    print(client)
    client.help("dataselect")


def get_data(start_date, end_date, station, network, path):
    mseed_path = f"{path}/{start_date}_{end_date}.mseed"
    xml_path = f"{path}/{start_date}_{end_date}.xml"

    # use local data if already downloaded
    if os.path.exists(mseed_path) and os.path.exists(xml_path):
        return (read(mseed_path), read_inventory(xml_path))

    st: obspy.Stream = client.get_waveforms(
        network=network,  # code réseau
        station=station,  # code station
        location="*",  # "*" = tous les codes de localisation, ou "00", "" etc.
        channel="*",  # canal : BHZ = large bande, composante verticale
        starttime=UTCDateTime(start_date),
        endtime=UTCDateTime(end_date),
    )  # type: ignore

    # get the instrument info
    inv: obspy.Inventory = client.get_stations(
        network=network,
        station=station,
        level="response",
    )  # type: ignore

    # local save
    st.write(mseed_path, format="MSEED")
    inv.write(xml_path, format="STATIONXML")

    return (st, inv)
