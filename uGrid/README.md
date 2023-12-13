#Getting Started

The uGrid tool has files that work together, or are inputs, and need to share a working folder for the tool to run smoothly

# uGrid Files

## Python Files: 
Each python file can be run seperately in an independent mode where the interconnecting variables are pre-set. To run the optimization
the Macro file is run.

Marco: macro_PC_3_alt.py: updated accept new technical tools code

Technical Tools: technical_tools_PC_3_alt.py: updated in 2019 to implement new dispatch algorithm

Economic Tools: economic_tools_PC_3.py 

Solar Calculations: solar_calcs_PC_3.py

## Input Files:
FullYearEnergy.xlsx: the forecasted load demand for during the coming night for the year (kWh/h)

8760: the load demand for each timestep for the year (kWh/h) for the area of interest

XXX_TMY.xlsx: the weather data for each timestep throughout the year for the area of interest, where XXX denotes the area (e.g MSU_TMY.xlsx for  - Maseru, Lesotho)

XXX_uGrid_input.xlsx: parameter inputs needed to run optimization where each sheet has the inputs for each of the python files

## Areas for Improvement:
Update of the PSO optimization for improved speed and accuracy

Create load forcasting tool

Add uncertainty in load forecasting and solar generation

## Prerequisite Python packages
The uGrid tool needs the following tools for it to work
    numpy scipy pandas scikit-learn xlsxwriter openpyxl requests
You can install them using pip install, either globally (not recommended), or in a virtual environment (recommended)
### Setting up a virtual environment
Install virtualenv
-- python -m pip install --user virtualenv
Set up a virtual environment
-- virtualenv {virtual environment name}
-- source {virtual environment name}/bin/activate
yada yada 
Install the dependencies 
-- pip install -r requirements.txt 

## Using the tool:
Ensure that all input files and code files are in the same working directory   
Use a terminal to navigate to the working directory   
Run:  python3 macro_PC_3_alt.py
      - You will be prompted to type in the area name (this could be a concession or any area of interest). Type it in and press enter.
