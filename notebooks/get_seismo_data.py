import pandas as pd
import re
import unicodedata

# ObsPy packages
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import kilometers2degrees

DATA = "../data"
NUM_DATA = 15

############# Function to get seismological data from different clients, for all the volcanoes studied ################

def get_seismo_data(start_date, t_before, t_after, client_name, network, station, channel_list, path, tag, volcano_name):

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

        start_str = str(start_date)
        start_date_short = start_str[:10]

        # local save
        st.write(f"{path}/data_eruptions_and_quiet/{start_date_short}_{volcano_name_clean}_{network}_{station}_{tag}.mseed", format="MSEED")
        inv.write(f"{path}/data_eruptions_and_quiet/{start_date_short}_{volcano_name_clean}_{network}_{station}_{tag}.xml", format="STATIONXML")

    except FDSNNoDataException as exc: # data error
        print(f"Data not found for volcano {volcano_name} (network: {network}, station: {station}) : {exc}")

    except Exception as exc: # other error
        print(f"Error for {volcano_name} (network: {network}, station: {station}) : {exc}")

    return(st, inv)


#######################################################################################################################
if __name__ == "__main__":

    path = DATA

    nb_hours_before = 12
    nb_hours_after = 24
    channel_list = "LH*,EH*,BH*"


    ####################################################################
    ######################## FOR ERUPTIONS #############################
    ########################################################################

    tag = 'erupt'

    df_eruptions = pd.read_csv(f'{path}/eruptions.csv', index_col=0)

    cpt_eruptions = 0

    for _, eruption in df_eruptions.sample(frac=1).iterrows() :
        if cpt_eruptions >= NUM_DATA : # if we have enough streams, we stop getting data
            break
        st, inv = get_seismo_data(start_date = eruption['Start Date'],
                        t_before = nb_hours_before,
                        t_after = nb_hours_after,
                        client_name = eruption['Client Name'],
                        network = eruption['Network'],
                        station = eruption['Station'],
                        channel_list = channel_list,
                        path = path,
                        tag = tag,
                        volcano_name = eruption['Volcano Name']
        )

        if st is not None :
            print(st, inv)
            cpt_eruptions+=1

    ########################################################################
    ######################## FOR NON-ERUPTIONS #############################
    ########################################################################

    tag = 'quiet'

    df_eruptions = pd.read_csv(f'{path}/eruptions.csv', index_col=0)

    # tri par volcan puis par date, et renumérotation 0..n-1
    df_eruptions = df_eruptions.sort_values(by=['Volcano Name', 'Start Date']).reset_index(drop=True)

    margin_before = nb_hours_before * 3600
    margin_after = nb_hours_after * 3600
    cpt_normal_stream = 0

    for i in df_eruptions.sample(frac=1).index:
        if cpt_normal_stream >= NUM_DATA:
            break

        eruption = df_eruptions.loc[i]
        volcano = eruption['Volcano Name']

        if pd.isna(eruption['End Date']):
            continue

        # cherche l'éruption suivante du MÊME volcan avec une date de début différente
        j = i + 1
        while (j < len(df_eruptions)
            and df_eruptions.loc[j, 'Volcano Name'] == volcano
            and df_eruptions.loc[j, 'Start Date'] == eruption['Start Date']):
            j += 1

        if j >= len(df_eruptions) or df_eruptions.loc[j, 'Volcano Name'] != volcano:
            continue  # pas d'éruption suivante pour ce volcan

        date_A = UTCDateTime(eruption['Start Date'])
        date_end_A = UTCDateTime(eruption['End Date'])
        date_B = UTCDateTime(df_eruptions.loc[j, 'Start Date'])
        date_mid = date_A + (date_B - date_A) / 2

        # toute la fenêtre doit tenir entre la fin de A et le début de B
        if date_mid - margin_before < date_end_A or date_mid + margin_after > date_B:
            print(f"Window too close to an eruption for {volcano}, skipped")
            continue

        st, inv = get_seismo_data(
            start_date = date_mid,
            t_before = nb_hours_before,
            t_after = nb_hours_after,
            client_name = eruption['Client Name'],
            network = eruption['Network'],
            station = eruption['Station'],
            channel_list = channel_list,
            path = path,
            tag = tag,
            volcano_name = volcano
        )

        if st is not None:
            cpt_normal_stream += 1
