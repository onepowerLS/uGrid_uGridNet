# Original Dispatch uGrid Codes

The macro and technical tool python files in this folder will solve using the original uGrid generator/battery/solar dispatch algorithm.
The updated "alt" code are in the above folder and were created in 2019.

The code in the above folder labeled "alt" has been updated from this original code in the following ways:

Macro:
~ update of technical tool code name to accept new technical_tools_PC_3_alt.py
~ removal of plotting functions, which are added to technical tools

Technical Tools: the main purpose of the update was to remove the operation states and directly calculate the power flows in and out
of devices. This significantly simplifies the code while maintaining the same results.
~ Updates to batt_bal: simplification and reduction of equations with no effective change to the output
~ GenControl is added
~ storage_levels is removed
~ getstate is removed: the states of operation are removed so power flows in and out of all devices are directly recorded. 
~ setvars is removed
~ plotting functions are added
