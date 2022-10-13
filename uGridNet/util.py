from __future__ import annotations

import datetime
import glob
import os
import re

import fiona
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
import pykml as pk

from pykml.factory import KML_ElementMaker as KML
from pykml.factory import GX_ElementMaker as GX

from lxml import etree

from constants import HOUSEHOLD_CURRENT, NOMINAL_MV_VOLTAGE, TRANSFORMER_PROPERTIES
from models import Pole, Line, SubNetwork, Branch, LineType, MVNetwork, GenerationSite, PoleType, Transformer

VISITED_POLES = []


def get_url(string_input: str) -> list[str]:
    return re.findall(r'(https?://\S+)', string_input)


def get_8760(village_name: str) -> str | None:
    filtered_list = glob.glob(f'{village_name}*8760*.xlsx')
    for f in filtered_list:
        if village_name in f and '8760' in f:
            return f
    return None


def create_pole_list_from_df(poleclasses_df: pd.DataFrame, droplines_df: pd.DataFrame) -> list[Pole]:
    poleclasses_df = poleclasses_df.dropna()
    pole_ids = poleclasses_df["ID"].tolist()
    pole_ids.sort()
    drop_pole_ids = droplines_df["DropPoleID"].tolist()
    # drop_pole_ids = droplines_df["Node 1"].tolist()
    poles = []
    for pole_id in pole_ids:
        index = poleclasses_df.ID[poleclasses_df.ID == pole_id].index.tolist()[0]
        num_of_connections = drop_pole_ids.count(pole_id)
        if num_of_connections is None:
            num_of_connections = 0
        current = drop_pole_ids.count(pole_id) * HOUSEHOLD_CURRENT
        latitude = poleclasses_df.loc[index, "GPS_Y"]
        longitude = poleclasses_df.loc[index, "GPS_X"]
        pole_type = PoleType[poleclasses_df.loc[index, "Type"]]
        pole = Pole(pole_id=pole_id,
                    connections=num_of_connections,
                    current=current,
                    voltage=0,
                    latitude=latitude,
                    longitude=longitude,
                    pole_type=pole_type
                    )
        if pole.pole_type == PoleType.MV:
            pole.voltage = NOMINAL_MV_VOLTAGE
        poles.append(pole)
    return poles


def get_pole_from_list(pole_id: str, poles: list[Pole]) -> Pole | None:
    try:
        pole = [p for p in poles if p.pole_id == pole_id][0]
        return pole
    except IndexError:
        return None


def get_next_pole(pole_id: str, branch_df: pd.DataFrame) -> list[str]:
    VISITED_POLES.append(pole_id)
    result_df = branch_df[
        # (branch_df["Node 2"] == pole_id) | (branch_df["Node 1"] == pole_id)]
        (branch_df["Pole_ID_To"] == pole_id) | (branch_df["Pole_ID_From"] == pole_id)]
    # next_pole_id = result_df["Node 2"].tolist() + result_df["Node 1"].tolist()
    next_pole_id = result_df["Pole_ID_To"].tolist() + result_df["Pole_ID_From"].tolist()
    next_pole_id = list(set(next_pole_id))
    if pole_id in next_pole_id:
        next_pole_id.remove(pole_id)
    return [p for p in next_pole_id if p not in VISITED_POLES]


def get_line_length(pole1_id: str, pole2_id: str, branch_df: pd.DataFrame) -> float:
    # result1 = branch_df.query(f'`Node 1` == "{pole1_id}" and `Node 2` == "{pole2_id}"')["Length"].tolist()
    result1 = branch_df.query(f'`Pole_ID_From` == "{pole1_id}" and `Pole_ID_To` == "{pole2_id}"')["adj_length"].tolist()
    # result2 = branch_df.query(f'`Node 2` == "{pole1_id}" and `Node 1` == "{pole2_id}"')["Length"].tolist()
    result2 = branch_df.query(f'`Pole_ID_To` == "{pole1_id}" and `Pole_ID_From` == "{pole2_id}"')["adj_length"].tolist()
    length = result1[0] if len(result1) > 0 else result2[0]
    return length


def generate_digraph_edges(first_pole_id: str, filtered_df: pd.DataFrame, poles: list[Pole],
                           line_type: LineType) -> list:
    next_poles = get_next_pole(first_pole_id, filtered_df)
    if len(next_poles) == 0:
        VISITED_POLES.append(first_pole_id)
        return []
    else:
        the_results = []
        for pole_2_id in next_poles:
            pole_1 = get_pole_from_list(pole_id=first_pole_id, poles=poles)
            pole_2 = get_pole_from_list(pole_id=pole_2_id, poles=poles)
            length = get_line_length(first_pole_id, pole_2_id, filtered_df)
            the_results.append((pole_1, pole_2, {"line": Line(line_type=line_type, voltage_drop=0, length=length)}))
            the_results += generate_digraph_edges(pole_2_id, filtered_df, poles, line_type)
            VISITED_POLES.append(first_pole_id)
        return the_results


def create_subnetworks_from_df(networklines_df: pd.DataFrame, poles: list[Pole]) -> list[SubNetwork]:
    networklines_df = networklines_df.dropna()
    # pole_ids_to = networklines_df["Node 2"].tolist()
    pole_ids_to = networklines_df["Pole_ID_To"].tolist()
    pole_ids_from = networklines_df["Pole_ID_From"].tolist()
    # pole_ids_from = networklines_df["Node 1"].tolist()
    unique_pole_ids = list(set(pole_ids_to + pole_ids_from))
    branch_names = [pole_id[:9] for pole_id in unique_pole_ids if pole_id[7] != "M"]
    branch_names = list(set(branch_names))
    branch_names.sort()

    subnetwork_names = list(set([branch_name[:-1] for branch_name in branch_names]))
    subnetwork_names.sort()
    print(subnetwork_names)
    subnetworks = [SubNetwork(name=subnetwork_name, branches=[]) for subnetwork_name in subnetwork_names]
    print(subnetworks)

    for branch_name in branch_names:
        VISITED_POLES.clear()
        branch_df = networklines_df[
            # networklines_df["Node 2"].str.contains(branch_name) | networklines_df["Node 1"].str.contains(
            networklines_df["Pole_ID_To"].str.contains(branch_name) | networklines_df["Pole_ID_From"].str.contains(
                branch_name)]
        transformer_pole_id_df = branch_df[
            # branch_df["Node 2"].str.endswith(branch_name[-2]) | branch_df["Node 1"].str.endswith(
            branch_df["Pole_ID_To"].str.endswith(branch_name[-2]) | branch_df["Pole_ID_From"].str.endswith(
                branch_name[-2])]

        try:
            pole_id_from_ = transformer_pole_id_df["Pole_ID_From"].values[0]
            # pole_id_from_ = transformer_pole_id_df["Node 1"].values[0]
        except IndexError:
            pole_id_from_ = None

        try:
            pole_id_to_ = transformer_pole_id_df["Pole_ID_To"].values[0]
            # pole_id_to_ = transformer_pole_id_df["Node 2"].values[0]
        except IndexError:
            pole_id_to_ = None

        transformer_pole_id = pole_id_from_ if pole_id_from_ is not None and pole_id_from_[
            -1].isalpha() else pole_id_to_
        edges = generate_digraph_edges(first_pole_id=transformer_pole_id, filtered_df=branch_df, poles=poles,
                                       line_type=LineType.LV)
        branch_graph = nx.DiGraph(edges)
        list(nx.topological_sort(branch_graph))
        branch = Branch(name=branch_name, graph=branch_graph)
        subnetwork = [subnet for subnet in subnetworks if subnet.name == branch_name[:-1]][0]
        subnetwork.branches.append(branch)
        subnetwork.transformer_pole = get_pole_from_list(transformer_pole_id, poles)
        subnetwork.transformer = Transformer(transformer_id=transformer_pole_id)
    return subnetworks


def create_mv_net_from_df(poleclasses_df: pd.DataFrame, networklines_df: pd.DataFrame, poles: list[Pole]) -> MVNetwork:
    networklines_df = networklines_df.dropna()

    # Get gen site pole id
    gen_site_pole_df = poleclasses_df[poleclasses_df["distance_from_source"] == 0]
    gen_site_pole_id = gen_site_pole_df["ID"].tolist()[0]

    # Filter the networklines_df to get only MV lines
    mv_poles_df = networklines_df[networklines_df["Type"] == "MV"]

    # Create directed graph
    edges = generate_digraph_edges(first_pole_id=gen_site_pole_id, filtered_df=mv_poles_df, poles=poles,
                                   line_type=LineType.MV)
    mv_graph = nx.DiGraph(edges)
    list(nx.topological_sort(mv_graph))

    return MVNetwork(graph=mv_graph)


def output_voltage_to_gdf(poles: list[Pole]) -> gpd.GeoDataFrame:
    dataframe = pd.DataFrame.from_records([p.to_dict() for p in poles])
    #print(dataframe)
    geodataframe = gpd.GeoDataFrame(dataframe, geometry=gpd.points_from_xy(dataframe.Longitude, dataframe.Latitude))
    return geodataframe


def output_voltage_to_excel(dataframe: pd.DataFrame, village_name: str):
    now = datetime.datetime.now()
    dataframe.to_excel(f'outputs/{village_name}_Voltage{now.strftime("%Y_%m_%d_%H_%M_%S")}.xlsx')


def output_to_kml(kml_dir: str, filename: str, dataframe: gpd.GeoDataFrame):
    dataframe.rename({"pole_id": "name"})
    output_kml_file_path = os.path.join(kml_dir, filename)
    fiona.supported_drivers['KML'] = 'rw'
    with fiona.Env():
        dataframe.to_file(output_kml_file_path, driver='KML')

    # with open(output_kml_file_path) as f:
    #     root = pk.parse(f)

    # for pm in root.getroot().Documnet.Folder.Placemark:
    #     pm.insert(1,
    #               KML.Style(
    #                   KML.IconStyle(
    #                       KML.color(),
    #                       KML.scale(),
    #                       KML.Icon(
    #                           KML.href("")
    #                       )
    #                   ),
    #                   KML.LabelStyle(
    #                       KML.scale(0.8)
    #                   )
    #               ))
    #
    #     pm.Style = KML.Style(
    #         KML.LineStyle(
    #             KML.color(),
    #             KML.width(),
    #             GX.labelVisibility('1')
    #         ),
    #
    #     )
    # with open(output_kml_file_path, "w") as output:
    #     string = "<?xmk version=\"1.0\" encoding=\"utf-8\" ?>"
    #     string = string + etree.tostring(root, pretty_print=True).decode('utf8')
    #     output.write(string)


def filter_network_df(networklines_df: pd.DataFrame) -> pd.DataFrame:
    networklines_df = networklines_df.dropna()
    mv_line_df = networklines_df[networklines_df["Type"].str.contains("MV")]
    return mv_line_df


def get_gen_site(net_inputs_df: pd.DataFrame, village_id: str) -> GenerationSite:
    latitude = net_inputs_df['lat_Generation'][0]
    longitude = net_inputs_df['long_Generation'][0]
    return GenerationSite(gen_site_id=f"{village_id}_GEN_01", latitude=latitude, longitude=longitude)


def add_dropcon_ids(conn_input: pd.DataFrame, conn_output: pd.DataFrame, droplines: pd.DataFrame,
                    NetworkLines_output: pd.DataFrame):
    conn_ids = []

    # Add the connections IDs to ugridnet output file
    conn_input_copy = conn_input.filter(items=['Longitude', 'Latitude']).to_numpy()
    cic = conn_input_copy.copy()
    out_XY = np.zeros((len(conn_output), 2))
    out_XY[:, 0] = conn_output['GPS_X'].to_numpy()
    out_XY[:, 1] = conn_output['GPS_Y'].to_numpy()

    for v in out_XY:
        diff_dist = np.sqrt(np.power(cic[:, 0] - v[0], 2) + np.power(cic[:, 1] - v[1], 2))
        idx = np.where(diff_dist == diff_dist.min())[0]
        cic[idx, :] = np.array([180, 180])
        conn_ids.append(str(conn_input.loc[idx]['Name'].values[0]))
    conn_output['Name'] = conn_ids

    # Add the droplines  destination[i.e. connections] IDs
    consM = conn_output.filter(items=['index_x', 'index_y']).to_numpy()
    lineM = droplines.filter(items=['index_x_to', 'index_y_to']).to_numpy()
    consMstr = [str(int(consM[i, 0])) + str(int(consM[i, 1])) for i in range(len(consM))]
    lineMstr = [str(int(lineM[i, 0])) + str(int(lineM[i, 1])) for i in range(len(lineM))]
    survey_IDs = []
    for str_ in lineMstr:
        idx = consMstr.index(str_)
        survey_IDs.append(conn_output.iloc[idx]['Name'])
    droplines['DropConnID'] = survey_IDs
    village_number = droplines.iloc[1]['DropPoleID'][4] + droplines.iloc[2]['DropPoleID'][5]

    # Append the village Number to Subnetwork
    for i in range(len(droplines)):
        droplines.loc[i, 'SubNetwork'] = village_number + droplines.loc[i, 'SubNetwork']
    for i in range(len(NetworkLines_output)):
        if NetworkLines_output.loc[i, 'SubNetwork'] != '' and NetworkLines_output.loc[i, 'SubNetwork'] != 'M':
            NetworkLines_output.loc[i, 'SubNetwork'] = village_number + str(
                NetworkLines_output.loc[i, 'SubNetwork'])
            # export the modified data to the new excel file
    return conn_output, droplines, NetworkLines_output


def determine_transformer_size(subnetwork: SubNetwork) -> int | None:
    current = subnetwork.get_current()
    power = (current * HOUSEHOLD_CURRENT)/1000

    transformer = subnetwork.transformer

    sizes = TRANSFORMER_PROPERTIES.keys()

    for size in sizes:
        if (size - power) / power < 0.1:
            pass
        else:
            transformer.size = size
            return size
    return None
