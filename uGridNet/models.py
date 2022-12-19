# -*- coding: utf-8 -*-


"""
    The following module defines the objects that model some important entities in the network layout
"""
from enum import Enum

import networkx as nx

from constants import *


class ReticulationNetworkNode:
    """
        A
    """
    longitude = 0
    latitude = 0

    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude


class GenerationSite(ReticulationNetworkNode):
    def __init__(self, gen_site_id: str, latitude: float, longitude: float):
        super().__init__(latitude=latitude, longitude=longitude)
        self.gen_site_id = gen_site_id


class Connection(ReticulationNetworkNode):

    def __init__(self, site_number: str, connection_type: str, hh_pp: int, latitude: float = 0, longitude: float = 0):
        super().__init__(latitude, longitude)
        self.site_number = site_number
        self.connection_type = connection_type
        self.hh_pp = hh_pp


class PoleType(Enum):
    MV = "MV"
    LV = "LV"


class Pole(ReticulationNetworkNode):

    def __init__(self, pole_id: str, connections: int, current: float, voltage: float, latitude: float,
                 pole_type: PoleType,
                 longitude: float):
        super().__init__(latitude, longitude)
        self.pole_id = pole_id
        self.connections = connections
        self.current = current
        self.voltage = voltage
        self.pole_type = pole_type

    def to_dict(self):
        return {
            'ID': self.pole_id,
            'Type': self.pole_type.name,
            'numConnections': self.connections,
            'Current': self.current,
            'Voltage': round(self.voltage, 2),
            'Latitude': self.latitude,
            'Longitude': self.longitude
        }


class LineType(Enum):
    MV = "MV"
    LV = "LV"
    Dropline = "Dropline"


class Line:

    def __init__(self, length: float, voltage_drop: float, line_type: LineType):
        self.length = length
        self.voltage_drop = voltage_drop
        self.line_type = line_type


class ReticulationNetworkGraph:
    minimum_voltage: float
    graph: nx.DiGraph

    def get_length(self):
        lines = [data['line'] for pole1, pole2, data in list(self.graph.edges.data())]
        lengths = [line.length for line in lines]
        return sum(lengths)


class Branch(ReticulationNetworkGraph):
    minimum_voltage: float = NOMINAL_LV_VOLTAGE

    def __init__(self, name: str, graph: nx.DiGraph):
        self.name = name
        self.graph = graph

    def get_current(self) -> float:
        from network_calculations import calculate_current
        poles = list(self.graph.nodes)
        try:
            transformer_pole = poles[0]
            transformer_pole_current = calculate_current(transformer_pole, self)
            # print(f"Current at the transformer_pole is {transformer_pole_current}")
            return transformer_pole_current
        except IndexError:
            return 0

    def get_number_of_connections(self) -> int:
        poles = list(self.graph.nodes)
        connections = [pole.connections for pole in poles]
        return sum(connections)


class Transformer:
    size: float

    def __init__(self, transformer_id: str):
        self.transformer_id = transformer_id


class SubNetwork:
    transformer_pole: Pole
    transformer: Transformer

    def __init__(self, name: str, branches: list[Branch]):
        self.name = name
        self.branches = branches

    def get_current(self):
        branch_currents = [branch.get_current() for branch in self.branches]
        current = sum(branch_currents)
        try:
            self.transformer_pole.current = current
        except AttributeError:
            print("Transformer pole missing!")
        return current


class MVNetwork(ReticulationNetworkGraph):
    minimum_voltage: float = NOMINAL_MV_VOLTAGE

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph


class Network:
    graph: nx.DiGraph = nx.DiGraph()

    def __init__(self, name: str, mv_network: MVNetwork, subnetworks: list[SubNetwork]):
        self.name = name
        self.mv_network = mv_network
        self.subnetworks = subnetworks


class Cable:
    def __init__(self, size: float, cable_type: str, voltage_drop_constant: float, unit_cost: float):
        self.size = size
        self.cable_type = cable_type
        self.voltage_drop_constant = voltage_drop_constant
        self.unit_cost = unit_cost
