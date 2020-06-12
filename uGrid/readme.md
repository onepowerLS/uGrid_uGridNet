# uGrid Files

## Python Files: 
Each python file can be run seperately in an independent mode where the interconnecting variables are pre-set. To run the optimization
the Macro file is run.

Marco ~ macro_PC_3_alt.py: updated accept new technical tools code
Technical Tools ~ technical_tools_PC_3_alt.py: updated in 2019 to implement new dispatch algorithm
Economic Tools ~ economic_tools_PC_3.py 
Solar Calculations ~ solar_calcs_PC_3.py

## Input Files:
~ FullYearEnergy.xlsx: the forecasted load demand for during the coming night for the year (kWh/h)
~ LoadkW_MAK.xlsx: the load demand for each timestep for the year (kWh/h) for Ha Makebe (MAK)
~ MSU_TMY.xlsx: the weather data for each timestep throughout the year for Ha Makebe area (MSU - Maseru, Lesotho)
~ uGrid_input.xlsx: parameter inputs needed to run optimization where each sheet has the inputs for each of the python files

## Areas for Improvement:
~ Update of the PSO optimization for improved speed and accuracy
~ Create load forcasting tool
~ Add uncertainty in load forecasting and solar generation
