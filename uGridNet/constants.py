HOUSEHOLD_CURRENT = 2
LV_CABLES = [
    # {
    #     "Type": "LV ABC (2-Core)",
    #     "VoltageDropConstant": 0.00299,
    #     "Size": 25,
    # },
    {
        "Type": "LV ABC (2-Core)",
        "VoltageDropConstant": 0.00223,
        "Size": 35,
        "Cost": 2.1,
        "CurrentCapacity": 138
    },
    {
        "Type": "LV ABC (2-Core)",
        "VoltageDropConstant": 0.00165,
        "Size": 50,
        "Cost": 2.33,
        "CurrentCapacity": 168
    },
    {
        "Type": "LV ABC (2-Core)",
        "VoltageDropConstant": 0.00115,
        "Size": 70,
        "Cost": 2.72,
        "CurrentCapacity": 213
    },
    {
        "Type": "LV ABC (2-Core)",
        "VoltageDropConstant": 0.00084,
        "Size": 95,
        "Cost": 3.5,
        "CurrentCapacity":258
     },
    # {
    #     "Type": "LV ABC (2-Core)",
    #     "VoltageDropConstant": 0.00067,
    #     "Size": 120,
    # },
    # {
    #     "Type": "LV ABC (2-Core)",
    #     "VoltageDropConstant": 0.00055,
    #     "Size": 150,
    # },
]
NOMINAL_LV_VOLTAGE = 230
NOMINAL_MV_VOLTAGE = 11000
ASCR_FOX = {
    "Type": "ACSR FOX",
    "VoltageDropConstant": 0.7822 / 1000,
    "Size": 35,
    "Cost": 2.46
}
COSTS = [
    4843.61,
    2052.79,
    193.13,
    305.03,
    430.23,
    193.33,
    52.49,
    80.77,
    114.96,
    81.24,
    2.46,
    # 2.04,
    2.04,
    186.88,
]

REFERENCES = ['Assembly - Pole - MV - Start',
              'Assembly - Step - Down - Transformer',
              'Assembly - Pole - MV - Mid',
              'Assembly - Pole - MV - Bend <30',
              'Assembly - Pole - MV - Bend >30',
              'Assembly - Pole - MV - End',
              'Assembly - Pole - LV - Mid',
              'Assembly - Pole - LV - Bend <45',
              'Assembly - Pole - LV - Bend >45',
              'Assembly - Pole - LV - End',
              'Wire - MV',
              # 'Wire - LV',
              'Wire - LineDrop',
              'Assembly - Meter']

TRANSFORMER_PROPERTIES = {
    16: 300,
    25: 400,
    33: 500,
    50: 600,
}

# COST_DICT = {
#     references: REFERENCES,
#     quantities: 
# }

MERCATORS = {
    "S": "EPSG:22289",
    "N": "EPSG:32631",
}
