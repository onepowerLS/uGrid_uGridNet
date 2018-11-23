# -*- coding: utf-8 -*-
"""
uGrid "Macro" Code

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
minGen = 20
maxGen = 50
numInd = 20
X_tariff_multiplier = 0.005
stopLimit = 0.001
convergenceRequirement = 0.75*numInd
lowTestLim = 0.025
highTestLim = 0.05
roundDownSize = 0.05
C1 = 2
C2 = 2
CF = 1
W = 1
VF = 0.1
momentum = 0.95

#Parameter limits: Battery, PV
lower_bounds = [1,1]
upper_bounds = [10,5]


#Initialize matrixes and parameters
peakload = 40 #need to find where this is actually inputted
Parameters_test = np.zeros(2)
Parameters_dev = np.zeros(2)

Parameters = np.zeros((2,numInd,maxGen)) #this is the result for each parameter, for each month, for each individual
Propane_ec = np.zeros((numInd,maxGen))
Batt_kWh_tot_ec = np.zeros((numInd,maxGen))

#Create random initial guesses for Parameters
for i in range(2):
    for k in range(numInd):
        rn = np.random.uniform(lower_bounds[i],upper_bounds[i])
        if rn < roundDownSize: #Constraint for minimum sizes
            rn = 0
        Parameters[i,k,0] = np.copy(rn)
    
#Initialize Economic Parameters    
tariff = np.zeros((numInd,maxGen))
Batt_life_yrs = np.zeros((numInd,maxGen))
loanfactor=1
equity_debt_ratio=0
LEC = 0.1 #this is a starting point for LEC. This could potentially be done without a hill-climb and be directly solved

#Initialize Global Bests
#global best: best known postions ever, personal best: best known position for each particle(out of all generations), gbest_change: best in generation (out of all individuals)
#Global Best
gB_propane = 999
gB_tariff = 999
gB_tariff_plus = gB_tariff*(1+X_tariff_multiplier)
gB_parameters = np.zeros(2)

#gB_change (best individual in generation)
gB_change_propane = np.ones(maxGen)*999
gB_change_tariff = np.ones(maxGen)*999
gB_change_tariff_plus = gB_change_tariff*(1+X_tariff_multiplier)
gB_change_parameters = np.zeros((2,maxGen))

#Personal Best (best position for each particle)
pB_propane = np.ones(numInd)*999
pB_tariff = np.ones(numInd)*999
pB_parameters = np.ones((2,numInd))*999
pB_tariff_plus = np.ones(numInd)*999

#Initialize Velocities
#batt, PV, CSP, ORC, TES
VelMax = np.zeros(2)
VelMin = np.zeros(2)
VelMax = VF*(np.array(upper_bounds)-np.array(lower_bounds))
VelMin = -1*VelMax
V_gens = np.zeros((2,numInd,maxGen))


# Start Optimization Iterations
iteration = 0
t_diff = 999
match = 0
while iteration < maxGen-1 and t_diff > stopLimit and match < convergenceRequirement:
    match = 0
    for m in range(numInd):            
        #calculate technical parameters
        Propane_ec[m,iteration], DNI, Batt_kWh_tot_ec[m,iteration], Batt_SOC, Charge, State, LoadkW, genLoad, Inv_Batt_Dis_P, PV_Power, PV_Batt_Change_Power, dumpload, Batt_frac, Gen_Batt_Charge_Power, Genset_fuel, Fuel_kW = Tech_total(Parameters[0,m,iteration],Parameters[1,m,iteration])
        #don't need to save DNI, final, Batt_SOC, and Charge these are used to validate program
               
        LoanPrincipal, year, Cost, Revenue, CashonHand, Balance, M, O, tariff[m,iteration], Batt_life_yrs[m,iteration] = Econ_total(Propane_ec[m,iteration],Parameters[1,m,iteration]*peakload,Parameters[0,m,iteration]*peakload,Batt_kWh_tot_ec[m,iteration],loanfactor,equity_debt_ratio,LEC)
        #order of parameters: batt, PV, CSP, ORC, TES_ratio

        #Checking Outputs
        print "This individual's yearly propane and tariff is: " +str(Propane_ec[m,iteration]) + ", " +str(tariff[m,iteration])

        #Find generation best
        if tariff[m,iteration] < gB_change_tariff_plus[iteration] or (tariff[m,iteration] < gB_change_tariff[iteration] and Propane_ec[m,0] < gB_change_propane[iteration]):
            gB_change_tariff[iteration] = np.copy(tariff[m,iteration])
            gB_change_tariff_plus[iteration] = gB_change_tariff[iteration]*(1+X_tariff_multiplier)
            gB_change_propane[iteration] = np.copy(Propane_ec[m,iteration])
            gB_change_parameters[:,iteration] = np.copy(Parameters[:,m,iteration])
        #Find global best
        if tariff[m,iteration] < gB_tariff_plus or (tariff[m,iteration] < gB_tariff and Propane_ec[m,0] < gB_propane):
            gB_tariff = np.copy(tariff[m,iteration])
            gB_tariff_plus = gB_tariff*(1+X_tariff_multiplier)
            gB_propane = np.copy(Propane_ec[m,iteration])
            gB_parameters = np.copy(Parameters[:,m,iteration])
            #Saving plotting variables
            data_plot_variables = np.transpose([Batt_SOC, Charge, LoadkW, genLoad, Inv_Batt_Dis_P, PV_Power, PV_Batt_Change_Power, dumpload, Batt_frac, Gen_Batt_Charge_Power, Genset_fuel, Fuel_kW])
            gB_plot_variables = pd.DataFrame(data = data_plot_variables,columns=['Batt_SOC', 'Charge', 'LoadkW', 'genLoad', 'Inv_Batt_Dis_P', 'PV_Power', 'PV_Batt_Change_Power', 'dumpload', 'Batt_frac', 'Gen_Batt_Charge_Power', 'Genset_fuel', 'Fuel_kW'])
            gB_Cost = np.copy(Cost)         
        #Find personal best (done at the end of each iteration)
        for pB_iter in range(iteration):
            if tariff[m,iteration] < pB_tariff_plus[m] or (tariff[m,iteration] < pB_tariff[m] and Propane_ec[m,iteration] < pB_propane[m]):
                pB_tariff[m] = np.copy(tariff[m,iteration])
                pB_tariff_plus[m] = pB_tariff[m]*(1+X_tariff_multiplier)
                pB_propane[m] = np.copy(Propane_ec[m,iteration])
                pB_parameters[:,m] = np.copy(Parameters[:,m,iteration])

        #Calculate Next Gen
        V_gens[:,m,iteration+1] = W*V_gens[:,m,iteration] + C1*np.random.uniform(0,1)*(pB_parameters[:,m] - Parameters[:,m,iteration]) + C2*np.random.uniform(0,1)*(gB_parameters- Parameters[:,m,iteration])
        for s in range(2):
            if V_gens[s,m,iteration+1] > VelMax[s]:
                V_gens[s,m,iteration+1] = np.copy(VelMax[s])
            if V_gens[s,m,iteration+1] < VelMin[s]:
                V_gens[s,m,iteration+1] = np.copy(VelMin[s])
        # new microgrid parameters for the next generation
        # value from previous position plus some movement based on velocity & PSO parameters
        Parameters[:,m,iteration+1] = Parameters[:,m,iteration] + CF*V_gens[:,m,iteration+1]
        #// ceilings based on limits for specific microgrid parameters
        for q in range(2):
            if Parameters[q,m,iteration+1] > upper_bounds[q]:
                Parameters[q,m,iteration+1] = np.copy(upper_bounds[q])
            if Parameters[q,m,iteration+1] < lower_bounds[q]:
                Parameters[q,m,iteration+1] = np.copy(lower_bounds[q])
            if Parameters[q,m,iteration+1] < roundDownSize:
                Parameters[q,m,iteration+1] = 0
           
    #Stopping Criteria (once all individuals are calculated)
    if iteration < 30:
        testLim = np.copy(lowTestLim)
    else:
        testLim = np.copy(highTestLim)
    match = 0
    for ind in range(numInd):
        Parameters_test = (Parameters[:,ind,iteration] - gB_parameters)/(gB_parameters + 0.0001)
        Parameters_dev = np.zeros(2)
        for r in range(2):
             if abs(Parameters_test[r]) <= testLim:
                Parameters_dev[r] = 1
        PSO_match = sum(Parameters_dev)
        if PSO_match == 2:
            match = match + 1 #equivalent of previous code
    if match > convergenceRequirement:
        print "Stopping due to matching parameter values"
    
    #Checking for minimal change between bests
    if iteration >= 2:
        diff1 = abs(gB_change_tariff[iteration-2]-gB_change_tariff[iteration-1])
        diff2 = abs(gB_change_tariff[iteration-1] - gB_change_tariff[iteration])
        t_diff = diff1+diff2
        if t_diff < stopLimit:
            print "Stopping due to minimal change between global bests"
    
    
    #Print generation results
    print "Global Best Tariff "+ str(gB_tariff)
    print "Best Tariff in Generation " + str(gB_change_tariff[iteration])
    iteration += 1

end_time = time.time()
print "Time to complete generation is " + str(end_time - start_time)

#Best Solution Variables Saved
#gB_plot_variables
Total_Cost = sum(gB_Cost)
#test_pd['a'].plot(kind='line')
#test_pd.plot(kind='line')
#plt.savefig('Test.pdf')


#Plot Dispatch of resources powering battery

#Plot Dispatch of resources powering load

#Create print out of optimization results
#Best in generation and how it changes
#Final global best results
    
    
    
    



