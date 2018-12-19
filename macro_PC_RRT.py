# -*- coding: utf-8 -*-
"""
uGrid "Macro" Code with RRT instead of PSO

@author: Phy
"""

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from technical_tools_PC import Tech_total
from economic_tools_PC import Econ_total
import time

start_time = time.time()

#PSO Parameters
PSO_Parameters = pd.read_excel('uGrid_Input_RRT.xlsx', sheet_name = 'PSO')
deviation = PSO_Parameters['deviation'][0]
roundDownSize = PSO_Parameters['roundDownSize'][0]
MaxIter = PSO_Parameters['MaxIter'][0]

#Parameter limits: Battery, PV
#These will be changed so the solutions are more flexible for sites (not input)
#Could make these scaled off of load input
lower_bounds = [1,1]
upper_bounds = [10,5]


#Initialize matrixes and parameters
Parameters_test = np.zeros(2)
Parameters_dev = np.zeros(2)

tariff = np.zeros(MaxIter)
Batt_life_yrs = np.zeros(MaxIter)
Parameters = np.zeros((2,MaxIter)) #this is the result for each parameter, for each month, for each individual
Propane_ec = np.zeros(MaxIter)
Batt_kWh_tot_ec = np.zeros(MaxIter)

#Initialize Global Bests "Records"
#global best: best known postions ever, personal best: best known position for each particle(out of all generations), gbest_change: best in generation (out of all individuals)
#Global Best
#record temp and record
gB_tariff = 999
temp_tariff = 999

LoadKW_MAK = pd.read_excel('LoadKW_MAK.xlsx',index_col=None, header=None)
hmax = len(LoadKW_MAK)

#Only save Record
gB_propane = 999
gB_parameters = np.zeros(2)
gB_Cost = np.zeros(hmax)
gB_plot_variables = pd.DataFrame(data = np.zeros((hmax,13)) ,columns=['Batt_SOC', 'Charge', 'LoadkW', 'genLoad', 'Batt_Power_to_Load', 'Batt_Power_to_Load_neg', 'PV_Power', 'PV_Batt_Change_Power', 'dumpload', 'Batt_frac', 'Gen_Batt_Charge_Power', 'Genset_fuel', 'Fuel_kW'])

#Save record for each iteration to compare to PSO
recordRecord = np.zeros(MaxIter)

# Start Optimization Iterations
iteration = 0
t_diff = 999

for iteration in range(MaxIter):
    
    #Create random guesses for Parameters
    for i in range(2):
        rn = np.random.uniform(lower_bounds[i],upper_bounds[i])
        if rn < roundDownSize: #Constraint for minimum sizes
            rn = 0
        Parameters[i,iteration] = np.copy(rn)
    
    #calculate technical parameters
    Propane_ec[iteration], Gt_panel, Batt_kWh_tot_ec[iteration], Batt_SOC, Charge, State, LoadkW, genLoad, Inv_Batt_Dis_P, PV_Power, PV_Batt_Change_Power, dumpload, Batt_frac, Gen_Batt_Charge_Power, Genset_fuel, Fuel_kW, peakload, loadkWh = Tech_total(Parameters[0,iteration],Parameters[1,iteration])
    #don't need to save Gt_panel, final, Batt_SOC, and Charge these are used to validate program
           
    LoanPrincipal, year, Cost, Revenue, CashonHand, Balance, M, O, tariff[iteration], Batt_life_yrs[iteration] = Econ_total(Propane_ec[iteration],Parameters[1,iteration]*peakload,Parameters[0,iteration]*peakload,Batt_kWh_tot_ec[iteration],peakload,loadkWh)
    #order of parameters: batt, PV, CSP, ORC, TES_ratio
        
    #Checking Outputs
    print "This individual's yearly propane and tariff is: " +str(Propane_ec[iteration]) + ", " +str(tariff[iteration])

    #Find temp best
    if tariff[iteration] < gB_tariff*deviation:
        temp_tariff = np.copy(tariff[iteration])
    #Find global best 
    if temp_tariff < gB_tariff:
        gB_tariff = np.copy(temp_tariff)
        gB_propane = np.copy(Propane_ec[iteration])
        gB_parameters = np.copy(Parameters[:,iteration])
        #Saving plotting variables
        data_plot_variables = np.transpose([Batt_SOC, Charge, LoadkW, genLoad, -Inv_Batt_Dis_P, Inv_Batt_Dis_P, PV_Power, PV_Batt_Change_Power, dumpload, Batt_frac, Gen_Batt_Charge_Power, Genset_fuel, Fuel_kW])
        gB_plot_variables = pd.DataFrame(data = data_plot_variables,columns=['Batt_SOC', 'Charge', 'LoadkW', 'genLoad', 'Batt_Power_to_Load', 'Batt_Power_to_Load_neg', 'PV_Power', 'PV_Batt_Change_Power', 'dumpload', 'Batt_frac', 'Gen_Batt_Charge_Power', 'Genset_fuel', 'Fuel_kW'])
        gB_Cost = np.copy(Cost)
    
    #Save record for this iteration
    recordRecord[iteration] = np.copy(gB_tariff)
     
end_time = time.time()
total_time = end_time - start_time
print "Time to complete simulation is " + str(total_time)

#Best Solution Variables Saved
Total_Cost = sum(gB_Cost)
#add extra variables to solution output
gB_total_var =pd.DataFrame({'Total Cost':[Total_Cost],'Simulation_Time':[total_time],'Best Tariff':[gB_tariff], 'Best Propane':[gB_propane],'PVkW':[gB_parameters[0]],'BattkW':[gB_parameters[1]]})
gB_recordRecord = pd.DataFrame({'recordRecord':recordRecord})

#Print Results to Excel (Optimization and Solution Variables)
filename_xlsx = "uGrid_Output_"+PSO_Parameters['output_name'][0]+".xlsx"
writer = pd.ExcelWriter(filename_xlsx, engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
gB_plot_variables.to_excel(writer, sheet_name='Solution Output')
#gB_optimization_output.to_excel(writer, sheet_name='Optimization Output')
gB_total_var.to_excel(writer, sheet_name='Other Solution Output')
#Save recordRecord
gB_recordRecord.to_excel(writer, sheet_name='Record History')



# Close the Pandas Excel writer and output the Excel file.
writer.save()

#Plot power to the load and the load curve
gB_plot_variables.plot(y = ['LoadkW', 'genLoad', 'Batt_Power_to_Load', 'PV_Power','dumpload'], kind='line')
filename_Plot_LoadDispatch = "Power_to_Load_"+PSO_Parameters['output_name'][0]+".pdf"
plt.savefig(filename_Plot_LoadDispatch)
#Plot power to battery and battery SOC
gB_plot_variables.plot(y = ['Batt_SOC', 'Charge', 'PV_Batt_Change_Power','Batt_Power_to_Load_neg', 'Batt_frac', 'Gen_Batt_Charge_Power'], kind='line')
filename_Plot_BatteryDispatch = "Battery_to_Load_"+PSO_Parameters['output_name'][0]+".pdf"
plt.savefig(filename_Plot_BatteryDispatch)


    
    



