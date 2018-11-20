# -*- coding: utf-8 -*-
"""
uGrid "Macro" Code

@author: Phy
"""

from __future__ import division
import numpy as np
import pandas as pd
import math
from technical_tools_PC import Tech_total
from economic_tools_PC import Econ_total
import time

#PSO Parameters
minGen = 20
maxGen = 50
numInd = 20
numMonths = 12
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

lower_bounds = [1,1,0,0,1]
#batt, PV, CSP, ORC, TES
upper_bounds = [10,5,0,0,1]


#Initialize matrixes and parameters
peakload = 40 #need to find where this is actually inputted
Month_Parametric = 1 #this is as a parametric not sure how that works
numMonths = 12
Parameters_test = np.zeros(5)
Parameters_dev = np.zeros(5)

#Order of parameters: batt, PV, CSP, ORC, TES_ratio
Parameters = np.zeros((5,numInd,maxGen)) #this is the result for each parameter, for each month, for each individual
Propane = np.zeros((numMonths,numInd))
Batt_kWh_tot = np.zeros((numMonths,numInd))
Propane_ec = np.zeros((numInd,maxGen))
Batt_kWh_tot_ec = np.zeros((numInd,maxGen))

#Create random initial guesses for Parameters
for i in range(5):
    for k in range(numInd):
        rn = np.random.uniform(lower_bounds[i],upper_bounds[i])
        if rn < roundDownSize: #Constraint for minimum sizes
            rn = 0
        Parameters[i,k,0] = np.copy(rn)
    
#If no ORC, can't have CSP and TES
for l in range(numInd):
    if Parameters[3,l,0] == 0:
        Parameters[2,l,0] = 0
        Parameters[4,l,0] = 0 #This is TES_ratio, TESkWH is actually 0 though

#Initialize Economic Parameters    
tariff = np.zeros((numInd,maxGen))
Batt_life_yrs = np.zeros((numInd,maxGen))
loanfactor=1
equity_debt_ratio=0
LEC = 0.1 #not sure about this number

#Initialize Global Bests
#global best: best known postions ever, personal best: best known position for each particle(out of all generations), gbest_change: best in generation (out of all individuals)
#Global Best
gB_propane = 999
gB_tariff = 999
gB_tariff_plus = gB_tariff*(1+X_tariff_multiplier)
gB_parameters = np.zeros(5)

#gB_change (best individual in generation)
gB_change_propane = np.ones(maxGen)*999
gB_change_tariff = np.ones(maxGen)*999
gB_change_tariff_plus = gB_change_tariff*(1+X_tariff_multiplier)
gB_change_parameters = np.zeros((5,maxGen))

#Personal Best (best position for each particle)
pB_propane = np.ones(numInd)*999
pB_tariff = np.ones(numInd)*999
pB_parameters = np.ones((5,numInd))*999
pB_tariff_plus = np.ones(numInd)*999

#Initialize Velocities
#batt, PV, CSP, ORC, TES
VelMax = np.zeros(5)
VelMin = np.zeros(5)
VelMax = VF*(np.array(upper_bounds)-np.array(lower_bounds))
VelMin = -1*VelMax
V_gens = np.zeros((5,numInd,maxGen))


# Start Optimization Iterations
iteration = 0
diff = 999
match = 0
while iteration < maxGen-1 and diff > stopLimit and match < convergenceRequirement:
    start_time = time.time()
    match = 0
    for m in range(numInd):            
        for month in range(0,numMonths):
            #calculate technical parameters
            Propane[month,m], DNI, Batt_kWh_tot[month,m], final, Batt_SOC, Charge, State, LoadkW = Tech_total(Parameters[0,m,iteration],Parameters[1,m,iteration],Parameters[2,m,iteration],Parameters[3,m,iteration],Parameters[4,m,iteration],month+1)
            #don't need to save DNI, final, Batt_SOC, and Charge these are used to validate program
        Propane_ec[m,iteration] = sum(Propane[:,m])
        Batt_kWh_tot_ec[m,iteration] = sum(Batt_kWh_tot[:,m])
        
               
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
        #Find personal best (done at the end of each iteration)
        for pB_iter in range(iteration):
            if tariff[m,iteration] < pB_tariff_plus[m] or (tariff[m,iteration] < pB_tariff[m] and Propane_ec[m,iteration] < pB_propane[m]):
                pB_tariff[m] = np.copy(tariff[m,iteration])
                pB_tariff_plus[m] = pB_tariff[m]*(1+X_tariff_multiplier)
                pB_propane[m] = np.copy(Propane_ec[m,iteration])
                pB_parameters[:,m] = np.copy(Parameters[:,m,iteration])

        #Calculate Next Gen
        V_gens[:,m,iteration+1] = W*V_gens[:,m,iteration] + C1*np.random.uniform(0,1)*(pB_parameters[:,m] - Parameters[:,m,iteration]) + C2*np.random.uniform(0,1)*(gB_parameters- Parameters[:,m,iteration])
        for s in range(5):
            if V_gens[s,m,iteration+1] > VelMax[s]:
                V_gens[s,m,iteration+1] = np.copy(VelMax[s])
            if V_gens[s,m,iteration+1] < VelMin[s]:
                V_gens[s,m,iteration+1] = np.copy(VelMin[s])
        # new microgrid parameters for the next generation
        # value from previous position plus some movement based on velocity & PSO parameters
        Parameters[:,m,iteration+1] = Parameters[:,m,iteration] + CF*V_gens[:,m,iteration+1]
        #// ceilings based on limits for specific microgrid parameters
        for q in range(5):
            if Parameters[q,m,iteration+1] > upper_bounds[q]:
                Parameters[q,m,iteration+1] = np.copy(upper_bounds[q])
            if Parameters[q,m,iteration+1] < lower_bounds[q]:
                Parameters[q,m,iteration+1] = np.copy(lower_bounds[q])
            if Parameters[q,m,iteration+1] < roundDownSize:
                Parameters[q,m,iteration+1] = 0
        if Parameters[3,m,iteration+1] == 0:
            Parameters[2,m,iteration+1] = 0
            Parameters[4,m,iteration+1] = 0
            
            
    #Stopping Criteria (once all individuals are calculated)
    if iteration < 30:
        testLim = np.copy(lowTestLim)
    else:
        testLim = np.copy(highTestLim)
    match = 0
    for ind in range(numInd):
        Parameters_test = (Parameters[:,ind,iteration] - gB_parameters)/(gB_parameters + 0.0001)
        Parameters_dev = np.zeros(5)
        for r in range(5):
             if abs(Parameters_test[r]) <= testLim:
                Parameters_dev[r] = 1
        PSO_match = sum(Parameters_dev)
        if PSO_match == 5:
            match = match + 1 #equivalent of previous code
    if match > convergenceRequirement:
        print "Stopping due to matching parameter values"


    
    #Checking for minimal change between bests
    if iteration >= 2:
        diff = gB_change_tariff[iteration-2]-gB_change_tariff[iteration]
        if diff < stopLimit:
            print "Stopping due to minimal change between global bests"
    
    
    #Print generation results
    print "Global Best Tariff "+ str(gB_tariff)
    print "Best Tariff in Generation " + str(gB_change_tariff[iteration])
    iteration += 1
    end_time = time.time()
    print "Time to complete generation is " + str(end_time - start_time)
    
    
    
    



