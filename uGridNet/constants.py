import os
import sys

try:
    CONCESSION: str = sys.argv[1]
except IndexError:
    CONCESSION: str = input("CONCESSION: ")

try:
    VILLAGE_NUMBER: str = sys.argv[2]
except IndexError:
    VILLAGE_NUMBER: str = input("VILLAGE_NUMBER: ")
if "C" in VILLAGE_NUMBER:
    VILLAGE_NAME: str | None = None
else:
    try:
        VILLAGE_NAME: str = sys.argv[3]
    except IndexError:
        VILLAGE_NAME: str = input("VILLAGE NAME:  ")

VILLAGE_ID: str = f"{CONCESSION}_{VILLAGE_NUMBER}" if (VILLAGE_NUMBER is not None) else f"{CONCESSION}"
FULL_VILLAGE_NAME: str = f"{VILLAGE_ID}_{VILLAGE_NAME}" if (VILLAGE_NAME is not None) else f"{VILLAGE_ID}"

HOUSEHOLD_CURRENT = 2
AVAILABLE_CABLE_SIZES = [35, 50, 70]
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
    # {
    #     "Type": "LV ABC (2-Core)",
    #     "VoltageDropConstant": 0.00084,
    #     "Size": 95,
    # },
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
inputs_is_village_name = input("Is the inputs folder the name of village?[Y|N]:  ")
if inputs_is_village_name.upper() == "Y":
    INPUT_DIRECTORY = FULL_VILLAGE_NAME
else:
    INPUT_DIRECTORY = input("Input Directory: ")

OUTPUT_DIRECTORY = f"outputs/{FULL_VILLAGE_NAME}"
if not os.path.exists(OUTPUT_DIRECTORY):
    os.mkdir(OUTPUT_DIRECTORY)
# COST_DICT = {
#     references: REFERENCES,
#     quantities: 
# }
