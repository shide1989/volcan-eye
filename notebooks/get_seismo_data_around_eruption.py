import pandas as pd
import re
import unicodedata

# ObsPy packages
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import kilometers2degrees


############# Function to get seismological data from different clients, for all the volcanoes studied ################
def get_seismo_data(start_date, t_before, t_after, client_name, network, station, channel_list, path, volcano_name):

    d = UTCDateTime(start_date)
    start = d - t_before*3600
    end = d + t_after*3600

    print(d)

    client = Client(client_name, debug=True)

    st = None # empty stream
    inv = None # empty inventory (instrument)

    try:
        st = client.get_waveforms(
                network=network,           # code réseau
                station=station,         # code station
                location="*",          # "*" = tous les codes de localisation, ou "00", "" etc.
                channel=channel_list,         # canal
                starttime=start,
                endtime=end
            )

        # get the instrument info
        inv = client.get_stations(network=network, station=station, level="response")

        print(f"Data gotten for {volcano_name} (network: {network}, station: {station}).")

        # clean the volcano name
        volcano_name_clean = re.sub(
            r"[^A-Za-z0-9-]",                                  # 4. retire tout sauf lettres, chiffres et tirets
            "",
            re.sub(
                r"\s+", "-",                                   # 3. espaces -> tiret (plusieurs espaces = un seul tiret)
                unicodedata.normalize("NFKD", volcano_name)    # 1. décompose les accents (é -> e + ´)
                .encode("ascii", "ignore")                     # 2. supprime les accents (non-ASCII)
                .decode("ascii")
                .strip()                                       #    retire les espaces en début/fin
            ),
        )

        # local save
        st.write(f"{path}/data_eruptions/{start_date}_{volcano_name_clean}_{network}_{station}.mseed", format="MSEED")
        inv.write(f"{path}/data_eruptions/{start_date}_{volcano_name_clean}_{network}_{station}.xml", format="STATIONXML")

    except FDSNNoDataException as exc: # data error
        print(f"Error for volcano {volcano_name} (network: {network}, station: {station}) : {exc}")

    except Exception as exc: # other error
        print(f"Error for {volcano_name} (network: {network}, station: {station}) : {exc}")

    return(st, inv)




#######################################################################################################################
if __name__ == "__main__":

    path = '/home/julie/code/JulieCornet/VolcanEye/data'
    nb_hours_before = 10
    nb_hours_after = 4
    channel_list = "LH*,EH*,BH*"

    df_eruptions = pd.read_csv(f'{path}/eruptions.csv')

    cpt_eruptions = 0

    for _, eruption in df_eruptions.sample(frac=1).iterrows() :
        if cpt_eruptions >= 10 : # if we enough eruptions, we stop getting data
            break
        st, inv = get_seismo_data(start_date = eruption['Date'],
                        t_before = nb_hours_before,
                        t_after = nb_hours_after,
                        client_name = eruption['Client Name'],
                        network = eruption['Network'],
                        station = eruption['Station'],
                        channel_list = channel_list,
                        path = path,
                        volcano_name = eruption['Volcano Name']
        )

        if st is not None :
            print(st, inv)
            cpt_eruptions+=1
