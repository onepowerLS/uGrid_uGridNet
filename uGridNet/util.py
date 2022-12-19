# -*- coding: utf-8 -*-

"""
This module contains helper and utility functions can be re-usable across the program
"""
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

from constants import *
from models import *

VISITED_POLES = []


def get_url(string_input: str) -> list[str]:
    """
    Extracts URLs from a string
    Args:
        string_input: The string to get a URL from

    Returns:
        list[str]: A list of URLs found in the string.
    """
    return re.findall(r'(https?://\S+)', string_input)


def get_8760(village_name: str) -> str | None:
    """
        Finds the exact name of the 8760 file in the input folder.

        The need for this function appeared because 8760 files were not saved with the same name convention

    Args:
        village_name: The village for which the 8760 file is searched

    Returns:
        str | None: the name of the 8760 file.
    """
    filtered_list = glob.glob(f'{village_name}*8760*.xlsx')
    for f in filtered_list:
        if village_name in f and '8760' in f:
            return f
    return None


def create_pole_list_from_df(poleclasses_df: pd.DataFrame, droplines_df: pd.DataFrame) -> list[Pole]:
    """
        Creates a list of pole object from poleclasses and droplines dataframes from the ClassifyNetwork function in
        uGridNet_runner
    Args:
        poleclasses_df: dataframe with pole details including coordinates and whether they are LV or MV poles
        droplines_df: dataframe with details of droplines including connections and poles they connect

    Returns:
        list[Pole] : a list of Pole of objects
    """
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
    """
        Gets a pole object with a matching id from a list of poles
    Args:
        pole_id: unique pole ID
        poles: list of poles in a network

    Returns:
        Pole: the pole with the ID provided
    """
    try:
        pole = [p for p in poles if p.pole_id == pole_id][0]
        return pole
    except IndexError:
        return None


def get_next_pole(pole_id: str, branch_df: pd.DataFrame) -> list[str]:
    """
    Finds the next pole by traversing the network branch from the transformer towards the connections
    Args:
        pole_id: a unique pole ID
        branch_df: A subset of the networklines dataframe filtered by branch

    Returns:
        list[str]: IDs of poles that are directly connected downstream from pole_id
    """
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
    """
        Gets the length of a specific line from the branch dataframe
    Args:
        pole1_id: An ID if a pole that is connected by the line in qyuestion
        pole2_id: An ID of another pole that connects the line
        branch_df: A subset of the networklines dataframe filtered by branch

    Returns:
        float: The length in meters of the line
    """
    # result1 = branch_df.query(f'`Node 1` == "{pole1_id}" and `Node 2` == "{pole2_id}"')["Length"].tolist()
    result1 = branch_df.query(f'`Pole_ID_From` == "{pole1_id}" and `Pole_ID_To` == "{pole2_id}"')["adj_length"].tolist()
    # result2 = branch_df.query(f'`Node 2` == "{pole1_id}" and `Node 1` == "{pole2_id}"')["Length"].tolist()
    result2 = branch_df.query(f'`Pole_ID_To` == "{pole1_id}" and `Pole_ID_From` == "{pole2_id}"')["adj_length"].tolist()
    length = result1[0] if len(result1) > 0 else result2[0]
    return length


def generate_digraph_edges(first_pole_id: str, filtered_df: pd.DataFrame, poles: list[Pole],
                           line_type: LineType) -> list:
    """
        Creates edges of an nx.DiGraph to create a SubNetwork or Network
    Args:
        first_pole_id: The id of the root pole in the graph
        filtered_df:  A subset of the networklines dataframe
        poles: A list of pole objects
        line_type: LV or MV

    Returns:
`       list: list of edges of the graph
    """
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


def update_subnetwork_name(name: str) -> str:
    """
        Updates name from networklines dataframe to be used with subnetwork class
    Args:
        name: subnetwork name from networklines

    Returns:
        new subnetwork name
    """
    new_name = f"{VILLAGE_ID}_{name[-1]}"
    return new_name


def update_branch_name(branch_name: str, subnetwork_name: str) -> str:
    """
        Updates branch name from networklines dataframe to be used with subnetwork class
    Args:
        branch_name: branch name from networklines
        subnetwork_name: corresponding subnetwork
    Returns:
        new subnetwork name
    """
    new_name = f"{VILLAGE_ID}_{subnetwork_name[-1]}{branch_name}"
    return new_name


def create_subnetworks_from_df(networklines_df: pd.DataFrame, poles: list[Pole]) -> list[SubNetwork]:
    """
     Creates model.SubNetwork objects for the entire network
    Args:
        networklines_df: dataframe with all nodes of a network
        poles: a list pole of models.Pole objects of a network

    Returns:
        list[SubNetwork]: A list of models.SubNetwork objects of a network.
    """
    networklines_df = networklines_df.dropna()

    subnetwork_column = networklines_df["SubNetwork"].to_list()
    branch_column = networklines_df["Branch"].to_list()

    branch_names = []
    for i in range(len(branch_column)):
        branch_names.append(update_branch_name(branch_column[i], subnetwork_column[i]))
    branch_names = list(set(branch_names))
    branch_names = [b for b in branch_names if b[-2] != "M"]
    branch_names.sort()

    subs = list(set(subnetwork_column))
    subs.sort()

    subnetwork_names = list(set([update_subnetwork_name(s) for s in subs]))
    subnetwork_names = [s for s in subnetwork_names if not s.endswith("M")]
    # subnetwork_names = list(set([branch_name[:-1] for branch_name in branch_names]))
    # networklines_df.to_excel(f"{len(subnetwork_names)}_{datetime.datetime.now()}_transformers.xlsx")
    subnetwork_names.sort()
    subnetworks = [SubNetwork(name=subnetwork_name, branches=[]) for subnetwork_name in subnetwork_names]

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
    """
        Creates an MVNetwork for the network
    Args:
        poleclasses_df:  dataframe with pole details including coordinates and whether they are LV or MV poles
        networklines_df: dataframe with all nodes of a network
        poles: a list of models.Pole objects in the network

    Returns:
        MVNetwork: the MV Network
    """
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
    """
        Creates a geoDataFrame of poles with their voltages
    Args:
        poles: a list of models.Pole objects

    Returns:
        gpp.GeoDataFrame: a GeoPandas Dataframe with pole ids, voltages and other details
    """
    dataframe = pd.DataFrame.from_records([p.to_dict() for p in poles])
    geodataframe = gpd.GeoDataFrame(dataframe, geometry=gpd.points_from_xy(dataframe.Longitude, dataframe.Latitude))
    return geodataframe


def output_voltage_to_excel(dataframe: pd.DataFrame, village_name: str) -> str:
    """
        Exports voltage dataframe to a timestamped Excel file
    Args:
        dataframe: Voltage dataframe
        village_name: name of village

    Returns:
        str: The path of the Excel file output
    """
    now = datetime.datetime.now()
    output_path = f'{OUTPUT_DIRECTORY}/{village_name}_Voltage{now.strftime("%Y_%m_%d_%H_%M_%S")}.xlsx'
    dataframe.to_excel()
    return output_path


def output_to_kml(kml_dir: str, filename: str, dataframe: gpd.GeoDataFrame) -> str:
    """
        Export network layout as KML file
    Args:
        kml_dir: Directory to save KML files
        filename: Output filename
        dataframe: a GeoPandas dataframe

    Returns:
        str: Full path of the KML file
    """
    dataframe.rename({"pole_id": "name"})
    output_kml_file_path = os.path.join(kml_dir, filename)
    fiona.supported_drivers['KML'] = 'rw'
    with fiona.Env():
        dataframe.to_file(output_kml_file_path, driver='KML')
    return output_kml_file_path

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


def filter_network_df_by_type(networklines_df: pd.DataFrame, type: PoleType) -> pd.DataFrame:
    """
        Filter network dataframe by voltage size
    Args:
        networklines_df: dataframe with all network nodes' IDs
        type: MV or LV

    Returns:
        pd.DataFrame: the filtered dataframe
    """
    networklines_df = networklines_df.dropna()
    mv_line_df = networklines_df[networklines_df["Type"].str.contains(type.value)]
    return mv_line_df


def get_gen_site(net_inputs_df: pd.DataFrame, village_id: str) -> GenerationSite:
    """
        Gets the gen_site_id from inputs
    Args:
        net_inputs_df: network inputs dataframe
        village_id: the village id

    Returns:
        GenerationSite: models.GenerationSite object
    """
    latitude = net_inputs_df['lat_Generation'][0]
    longitude = net_inputs_df['long_Generation'][0]
    return GenerationSite(gen_site_id=f"{village_id}_GEN_01", latitude=latitude, longitude=longitude)


def add_dropcon_ids(conn_input: pd.DataFrame, conn_output: pd.DataFrame, droplines: pd.DataFrame,
                    NetworkLines_output: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
        Updates uGridNet output to include connection IDs for droplines
    Args:
        conn_input: dataframe with connections from the input file.
        conn_output: dataframe with connections from ClassifyNetwork function
        droplines: dataframe with droplines properties from the ClassifyNetwork function
        NetworkLines_output: dataframe with networklines from the ClassifyNetwork function

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: updated connections, droplines and networklines outputs
    """
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
    """
        Evaluates the best size of transformer to use in a subnetwork
    Args:
        subnetwork: The subnetwork to be evaluated

    Returns:
        int: The size in kVA of the selected transformer
    """
    current = subnetwork.get_current()
    power = (current * HOUSEHOLD_CURRENT) / 1000

    transformer = subnetwork.transformer

    sizes = TRANSFORMER_PROPERTIES.keys()

    for size in sizes:
        if (size - power) / power < 0.1:
            pass
        else:
            transformer.size = size
            return size
    return None


if __name__ == "__main__":
    net_lines = pd.read_excel("4_2022-12-15 14:05:31.651500_transformers.xlsx")
    create_subnetworks_from_df(networklines_df=net_lines, poles=[])
