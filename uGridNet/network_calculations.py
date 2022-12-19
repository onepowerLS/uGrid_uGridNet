from __future__ import annotations
from ast import Index
from collections import Counter

import datetime

import numpy as np
import pandas as pd

from constants import *
from models import *
from util import *


def calculate_current(pole: Pole, branch: Branch) -> float:
    """
    Calculates the current at a particular pole assumning all connections are consuming HOUSEHOLD CURRENT
    Args:
        pole: the pole in question
        branch: the branch the pole belongs to

    Returns:
        The total current through the pole
    """
    connected_poles = list(branch.graph.successors(pole))
    current = pole.connections * HOUSEHOLD_CURRENT
    if len(connected_poles) == 0:
        pole.current = current

    else:
        connected_poles_currents = [calculate_current(connected_pole, branch) for connected_pole in connected_poles]
        sum_of_currents = sum(connected_poles_currents)
        pole.current = sum_of_currents + current
    return pole.current


def calculate_voltage_drop_for_all_lines(branch: Branch, cable: Cable):
    """
    Calculates the voltage drop in a branch for a chosen cable
    Args:
        branch: The branch in question
        cable: The cable chosen

    Returns:
        None
    """
    for pole1, pole2, data in list(branch.graph.edges.data()):
        lv_line = data["line"]
        calculate_voltage_drop(pole2, lv_line, cable)


def calculate_voltage_drop(destination_pole: Pole, line: Line, cable: Cable) -> float:
    """
     Calculates the voltage drop between two poles for a given cable size
    Args:
        destination_pole: The pole the line is headed to starting from the transformer
        line: The line connecting the poles
        cable: The cable used in the line

    Returns:
        The voltage drop between the poles
    """
    line.voltage_drop = line.length * cable.voltage_drop_constant * destination_pole.current
    return line.voltage_drop


def calculate_voltage_for_all_poles(ret_graph: ReticulationNetworkGraph) -> float:
    """
        Calculates the voltage for all poles in a graph
    Args:
        ret_graph:

    Returns:
        The minimum voltage in the graph
    """
    terminal_poles = [pole for pole in ret_graph.graph.nodes() if
                      ret_graph.graph.in_degree(pole) != 0 and ret_graph.graph.out_degree(pole) == 0]
    for tp in terminal_poles:
        calculate_voltage(tp, ret_graph)
    terminal_poles.sort(key=lambda p: p.voltage)
    try:
        pole_with_min_voltage = terminal_poles[0]
        voltage = pole_with_min_voltage.voltage
        ret_graph.minimum_voltage = voltage
        return voltage
    except IndexError:
        return 0


def calculate_voltage(pole: Pole, branch: Branch) -> float:
    """

    Args:
        pole:
        branch:

    Returns:

    """
    parents = list(branch.graph.predecessors(pole))
    if len(parents) == 0:
        pole.voltage = NOMINAL_LV_VOLTAGE
        return pole.voltage
    else:
        parent = parents[0]
        data = branch.graph.get_edge_data(parent, pole)
        line = data["line"]
        pole.voltage = calculate_voltage(parents[0], branch) - line.voltage_drop
        return pole.voltage


def determine_cable_size(branch: Branch) -> tuple[Cable | None, float | None]:
    """

    Args:
        branch:

    Returns:

    """
    cable_choices = [
        Cable(size=cable["Size"], cable_type=cable["Type"], voltage_drop_constant=cable["VoltageDropConstant"],
              unit_cost=cable["Cost"])
        for cable in LV_CABLES
    ]
    try:
        branch.get_current()
    except IndexError:
        return None, None
    for choice in cable_choices:
        calculate_voltage_drop_for_all_lines(branch, choice)
        voltage = calculate_voltage_for_all_poles(branch)
        if voltage >= 214:
            return choice, voltage
    return None, None


def network_calculations(
        networklines_df: pd.DataFrame,
        poleclasses_df: pd.DataFrame,
        droplines_df: pd.DataFrame,
        poles_list: list[Pole]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Args:
        networklines_df: properties of MV and LV lines in the network
        poleclasses_df: properties of the poles
        droplines_df: properties of lines from poles to service connections
        poles_list: a list of the poles as models.Pole objects
    Returns:
        Two dataframes, cable choices and network costs
    """

    results_list = []
    cable_choices = {}
    columns = ["SubNetwork", "Branch", "Connections", "LineType", "CableType", "NominalVoltage",
               "MinimumVoltage", "Length", "Current"]  # columns of the cable choices dataframe
    for i in AVAILABLE_CABLE_SIZES:
        columns.append(f"CableSize {i}")
    v_drop_df = pd.DataFrame(columns=columns)  # initialize cable choices dataframe

    # MV network calculation
    mv_network = create_mv_net_from_df(poleclasses_df=poleclasses_df, networklines_df=networklines_df, poles=poles_list)
    mv_results = {
        "SubNetwork": "M",
        "Branch": 1,
        "Connections": "N/A",
        "LineType": "MV",
        "CableType": "FOX ACSR",
        "Length": mv_network.get_length(),
        "CableSize 35": "N/A",
        "CableSize 70": "N/A",
        "CableSize 50": "N/A",
        "NominalVoltage": 11000,
        "MinimumVoltage": mv_network.minimum_voltage,
        "Current": "N/A"
    }

    # LV network calculation
    subnetworks_list = create_subnetworks_from_df(networklines_df, poles_list)
    for sub in subnetworks_list:
        for b in sub.branches:
            branch_current = b.get_current()
            length = b.get_length()
            cable, least_voltage = determine_cable_size(b)
            result_info = {
                "SubNetwork": sub.name[-1],
                "Branch": b.name[-1],
                "Connections": b.get_number_of_connections(),
                "LineType": "LV",
                "NominalVoltage": NOMINAL_LV_VOLTAGE,
                "MinimumVoltage": least_voltage,
                "Length": length,
                "Current": branch_current}
            try:
                result_info["CableType"] = cable.cable_type
                for i in AVAILABLE_CABLE_SIZES:
                    if i >= cable.size:
                        result_info[f"CableSize {i}"] = "Pass"
                    else:
                        result_info[f"CableSize {i}"] = "Fail"
                results_list.append(result_info)

                # Add up the length of each cable_choice
                try:
                    cable_choices[cable] = cable_choices[cable] + length  # add if cable chosen in another branch
                except KeyError:
                    cable_choices[cable] = length

            # If the cable is None it means the network fails for all available cable choices.
            except AttributeError:
                result_info["CableType"] = "Fail"
                for i in AVAILABLE_CABLE_SIZES:
                    result_info[f"CableSize {i}"] = "Fail"
                results_list.append(result_info)
            sub.get_current()  # Update the current calculation for the subnetwork

    results_list.append(mv_results)  # Add MV results to the bottom of the list

    v_drop_df = v_drop_df.append(results_list, ignore_index=True, sort=False)  # Add the results to the dataframe

    # Transformer Calculations
    #
    # transformer_choices = [determine_transformer_size(subnet) for subnet in subnetworks_list]
    #
    # # Cost Calculations
    # transformer_counts = Counter(transformer_choices)
    # chosen_transformer_sizes = transformer_counts.keys()
    # transformer_costs = dict(zip(chosen_transformer_sizes, [None] * len(chosen_transformer_sizes)))
    # for c in transformer_costs:
    #     transformer_costs[c] = TRANSFORMER_PROPERTIES[c] * transformer_counts[c]

    cable_quantities = {}
    cable_costs = {}
    for cbl in cable_choices.keys():
        cbl_choice = f"{cbl.size} mm {cbl.cable_type}"
        try:
            cable_quantities[cbl_choice] = cable_quantities[cbl_choice] + cable_choices[cbl]
            # print(f'cable quantities {cable_quantities}')
        except KeyError:
            cable_quantities[cbl_choice] = cable_choices[cbl]
            # print(f'cable quantities after exception {cable_quantities}')
            cable_costs[cbl_choice] = cbl.unit_cost

    mv_ref = poleclasses_df[poleclasses_df.Type == 'MV']
    mv_ref = mv_ref[mv_ref.distance_from_source != 0]
    lv_ref = poleclasses_df[poleclasses_df.Type == 'LV']

    cable_references = list(cable_quantities.keys())
    # print(f'my cable references {cable_references}')

    references = REFERENCES + list(cable_quantities.keys())
    quantity = np.zeros(len(references))
    costs = COSTS + list(cable_costs.values())
    # print(f'costs {costs}, quantities {cable_quantities.values()}')
    # print(len(cable_references))
    for i in range(len(cable_quantities.values())):
        try:
            quantity[-1 - i] = list(cable_quantities.values())[-1 - i]
        except IndexError:
            pass

    quantity[0] = 1

    num_transformers = 0
    for item in mv_ref.ID.values:
        if item[-1].isalpha() == True:
            num_transformers += 1
    quantity[1] = num_transformers

    mv_pole_counts = mv_ref.value_counts(subset=['AngleClass'])
    try:
        quantity[2] = mv_pole_counts['mid_straight']
    except KeyError:
        pass
    try:
        quantity[3] = mv_pole_counts['mid_less_30']
    except KeyError:
        pass
    try:
        quantity[4] = mv_pole_counts['mid_over_30']
    except KeyError:
        pass
    try:
        quantity[5] = mv_pole_counts['terminal']
    except KeyError:
        pass

    lv_pole_counts = lv_ref.value_counts(subset=['AngleClass'])
    try:
        quantity[6] = lv_pole_counts['mid_straight']
    except KeyError:
        pass
    try:
        quantity[7] = lv_pole_counts['mid_less_45']
    except KeyError:
        pass
    try:
        quantity[8] = lv_pole_counts['mid_over_45']
    except KeyError:
        pass
    try:
        quantity[9] = lv_pole_counts['terminal']
    except KeyError:
        pass

    # Lines length
    mvnet = networklines_df[networklines_df.Type == 'MV']
    quantity[10] = mvnet.adj_length.values.sum()

    # Line drop to households
    quantity[11] = droplines_df.Linedrop.values.sum()
    quantity[12] = len(droplines_df)

    try:
        temp_df = pd.DataFrame({'Part Reference': references,
                                'Qty': quantity,
                                'Price (USD)': costs})
    except ValueError:
        cutoff = len(costs) - len(quantity)
        costs = costs[:-cutoff]
        temp_df = pd.DataFrame({'Part Reference': references,
                                'Qty': quantity,
                                'Price (USD)': costs})

    costs_df = temp_df[['Part Reference', 'Qty', 'Price (USD)']]
    costs_df['Line Total (USD)'] = costs_df['Qty'] * costs_df['Price (USD)']

    return v_drop_df, costs_df


if __name__ == "__main__":
    # file = "RIB_SC_uGridPlan.xlsx"\
    village = "RIB_86"
    # file = f"outputs/{village}/{village}/{village}/SEB_07_Moseneke_20220429_0907_uGrid.xlsx"
    file = "C:/Users/ONEPOWER ADMIN/PycharmProjects/uGrid_uGridNet/uGridNet/outputs/RIB_86_Ha_Nthonyana/RIB_86_Ha_Nthonyana/20220617_1155_RIB_uGrid_86_.xlsx"
    networklines = pd.read_excel(file, sheet_name="NetworkLength")
    poleclasses_df = pd.read_excel(file, sheet_name="PoleClasses")
    droplines = pd.read_excel(file, sheet_name="DropLines")
    net_inputs = pd.read_excel(
        "C:/Users/ONEPOWER ADMIN/PycharmProjects/uGrid_uGridNet/uGridNet/outputs/RIB_86_Ha_Nthonyana/RIB_86_Ha_Nthonyana" + '_uGrid_Input.xlsx',
        sheet_name='Net')

    poles = create_pole_list_from_df(poleclasses_df, droplines)
    #
    #
    vdrop, costs = network_calculations(networklines_df=networklines, poleclasses_df=poleclasses_df, poles_list=poles,
                                        droplines_df=droplines)

    # print(costs)
    costs.to_excel("Costs1.xlsx")
    # now = datetime.datetime.now()
    # dataframe = output_voltage_to_gdf(poles)
    # output_voltage_to_excel(dataframe, village)
    # # output_to_kml(".", "voltage_poles.kml", df)
    # results.to_excel(
    #     f'outputs/{village}_Vdrop{now.strftime("%Y_%m_%d_%H_%M_%S")}.xlsx')
