"""
Tech Tools

This contains all the technical functions which are called by the macro file
for the uGrid tool. 

@author: Phylicia Cicilio
"""

from __future__ import division
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from full_year_energy import * 
from constants import SITE_NAME
import glob


# from numba import jit

## Battery Calculations ========================================================
def batt_calcs(timestep, BattkWh, T_amb, Batt_SOC_in, Limit_charge, Limit_discharge):
    # This performs the battery performance calculations based on temperature and
    # the limits of the battery.
    # The output is the available charge and discharge amounts and the total
    # freespace in the battery. All these values are positive, this will be adjusted
    # for power flow negative and positive flow from the battery in the GenControl
    # function.

    # timestep = 1
    Self_discharge = timestep * 3600 * BattkWh * 3.41004573E-09 * np.exp(
        0.0693146968 * T_amb)  # "Effect of temperature on self discharge of the battery bank from 5% per month at 25C and doubling every 10C"
    Batt_SOC_in = Batt_SOC_in - Self_discharge
    SOC_frac = 0.711009126 + 0.0138667895 * T_amb - 0.0000933332591 * T_amb ** 2  # eta_{batt,temp}  #"Effect of temperature on the capacity of the battery bank from IEEE 485 Table 1"
    Batt_cap = BattkWh * SOC_frac
    high_trip = 0.95 * Batt_cap
    freespace_charge = high_trip - Batt_SOC_in
    if freespace_charge < 0:  # ensure not reporting negative amount of available space
        freespace_charge == 0
    low_trip = 0.05 * Batt_cap
    freespace_discharge = Batt_SOC_in - low_trip

    # account for charging and discharging limitrs
    Batt_discharge = min(Limit_discharge, freespace_discharge)
    Batt_charge = min(Limit_charge, freespace_charge)

    return Batt_discharge, Batt_charge, freespace_discharge  # These values are all positive


##=============================================================================

## Generation Control =========================================================
def GenControl(P_PV, L, Batt_discharge, Batt_charge, freespace_discharge, genPeak, LoadLeft, Batt_SOC, dayhour):
    # This performs the generation and load balance control. The algorithm performed
    # here is shown in the uGrid User Guide.
    # The output is the power flows from the battery and propance generator, the
    # state of charge of the battery, and the amount of power dumped due to excess
    # solar PV generation.

    if P_PV > 0:
        if P_PV - L < 0:  # Not enough PV to power load, gen running, battery charging
            if freespace_discharge > LoadLeft and Batt_discharge > L - P_PV:  # and dayhour < 12: #if in the morning (batteries can discharge to meet load)
                P_gen = 0
                P_batt = L - P_PV
                P_dump = 0
            else:
                P_gen = min(genPeak, (L - P_PV + Batt_charge))
                P_batt = -(P_gen + P_PV - L)
                P_dump = 0
        else:  # More than enough PV for load, gen off, battery charging
            P_gen = 0
            P_batt = - min(P_PV - L, Batt_charge)
            P_dump = P_PV + P_batt - L
    else:
        if freespace_discharge > LoadLeft and Batt_discharge > L:  # enough charge to run through night and meet load demand, PV and gen off
            P_gen = 0
            P_batt = np.copy(L)
            P_dump = 0
        else:  # not enough charge, PV off, gen on, battery charging
            P_gen = min(genPeak, (L + Batt_charge))
            P_batt = -(P_gen - L)
            P_dump = 0

    Batt_SOC = Batt_SOC - P_batt  # update state of charge of the battery

    return P_batt, P_gen, Batt_SOC, P_dump


# ==============================================================================


## fuel_calcs FUNCTION =====================================================================================================
# @jit(nopython=True) # Set "nopython" mode for best performance
def fuel_calcs(genload, peakload, timestep):
    # "generator fuel consumption calculations based on efficiency as a function of power output fraction of nameplate rating"

    partload = genload / peakload
    Eta_genset = -0.00430876206 + 0.372448046 * partload - 0.174532718 * partload ** 2  # "Derived from Onan 25KY model"

    if Eta_genset < 0.02:
        Eta_genset = 0.02  # "prevents erroneous results at extreme partload operation"

    Fuel_kW = genload / Eta_genset  # "[kW]"

    # "Fuel consumption"
    Fuel_kJ = Fuel_kW * timestep * 3600  # "[J]"
    Fuel_kg = Fuel_kJ / 50340  # "[converts J to kg propane]"

    return Fuel_kW, Fuel_kg


##=============================================================================================================================


## operation FUNCTION =====================================================================================================
def operation(Batt_Charge_Limit, smart, PVkW, BattkWh, peakload, LoadKW_MAK, FullYearEnergy, MSU_TMY, Solar_Parameters,
              trans_losses):
    # This function cycles through the year of data provided to determine the power flows and propane consumption for each timestep in
    # the year by calling all of the previous functions and the solar python file.

    from solar_calcs_PC_3 import SolarTotal

    # This code can and should be cleaned up *****

    # Month$ was replaced with Month_

    timestep = 1  # "hours - initialize timestep to 10 minutes"

    # Find length of Load Dataset
    hmax = len(LoadKW_MAK)
    Hour = np.arange(hmax)

    # initialize arrays
    P_PV = np.zeros(hmax)  # "kW electrical"
    P_batt = np.zeros(hmax)
    P_gen = np.zeros(hmax)
    P_dump = np.zeros(hmax)
    Batt_SOC = np.zeros(hmax + 1)
    dayHour = np.zeros(hmax)
    LoadkW = np.zeros(hmax)
    loadLeft = np.zeros(hmax)

    # "Initialize variables"
    Propane_kg = 0
    Limit_discharge = BattkWh * Batt_Charge_Limit
    Limit_charge = BattkWh * Batt_Charge_Limit

    # " If battery bank is included in generation equipment "
    if BattkWh > 0:
        Batt_SOC[0] = BattkWh / 4  # " Initialize the Batt Bank to quarter full"
    else:
        Batt_SOC[0] = 0  # "There is no Batt Bank"

    Batt_kWh_tot = 0

    # Loop through each timestep in the year
    h = 0  # "initialize loop variable (timestep counter)"
    while h < hmax - 16:

        # "============================="
        # " Analyze external conditions "
        # "============================="

        # "Establish the time of day in hours"
        if Hour[h] > 24:  # "factor the timestep down to get the time of day"
            factor = math.floor(Hour[h] / 24)
            dayHour[h] = Hour[h] - factor * 24
        else:
            dayHour[h] = np.copy(Hour[h])  # "timestep is the time of day during the first day"

        if dayHour[h] == 24:
            dayHour[h] = 0

        if PVkW > 0:
            # " Assess solar availability "
            hrang, declin, theta, Gt, P_PV[h], T_amb = SolarTotal(MSU_TMY, Solar_Parameters['year'][0], Hour[h],
                                                                  Solar_Parameters['longitude'][0],
                                                                  Solar_Parameters['latitude'][0],
                                                                  Solar_Parameters['timezone'][0],
                                                                  Solar_Parameters['slope'][0],
                                                                  Solar_Parameters['azimuth'][0],
                                                                  Solar_Parameters['pg'][0], PVkW,
                                                                  Solar_Parameters['fpv'][0],
                                                                  Solar_Parameters['alpha_p'][0],
                                                                  Solar_Parameters['eff_mpp'][0],
                                                                  Solar_Parameters['f_inv'][0])
        else:
            T_amb = -9999
            P_PV[h] = 0

        # "LOAD (kW) for this timestep"
        # "calculate the row value in the Nkautest simulated load dataset (based on Hobhouse NRS) corresponding to the simulation timestep"
        loadcounter = Hour[0] / timestep + h
        LoadkW[h] = LoadKW_MAK.iloc[int(loadcounter), 0] * (
                    1 + trans_losses)  # "choose the reference load dataset" #calls from lookuptable. Need to load excel sheet of this data
        # LOOKUP['Table Name', Row, Column]=Value, Set the value of the cell at the specified row andcolumn of the specified Lookup table to the given value. Column must either be an integer, referringto the column number or a string constant or string variable that provides the name of the column.

        # "*******************************************************************************************************"
        # "Smart charging strategy - the generator will only continue charging the battery bank if the SOC of the bank "
        # "is less than the SOC which would permit inverter operation until dawn (when the PV can ideally be prioritized "
        # "to recharge the bank at a lower operating cost than genset charging)"
        # "*******************************************************************************************************"

        if smart == 1:
            # " estimate load remaining until dawn for nighttimes (set to zero for daytimes) "
            # if dayHour[h] > 17 or dayHour[h] < 7: #"determine the amount of load in kWh remaining from premidnight until dawn"
            #    loadLeft[h] = FullYearEnergy.iloc[int(loadcounter-1),0]*0.75/5  #"Modify Nkau for number of households in community of interest" # I added *1.2 to have a little more overshoot
            # the FullYearEnergy provides the amount of forecasted nighttime load at each hour through the night (most loadleft at beginning of night, least load left at end of night)
            # else:
            #    loadLeft[h]=0 #"daytime load prediction not part of algorithm"

            # Try considering max LoadLeft for the next 12 hours
            loadLeft[h] = max(list(FullYearEnergy[h:h + 6][0])) * 0.75 / 5 * 9
            #print(pd.DataFrame(loadLeft))
        else:
            loadLeft[h] = 10000 + 2 * BattkWh  # "number much higher than can possibly be stored in battery bank"
            # " forces genset to stay on (not smart) "

        Batt_discharge, Batt_charge, freespace_discharge = batt_calcs(timestep, BattkWh, T_amb, Batt_SOC[h],
                                                                      Limit_charge, Limit_discharge)
        P_batt[h], P_gen[h], Batt_SOC[h + 1], P_dump[h] = GenControl(P_PV[h], LoadkW[h], Batt_discharge, Batt_charge,
                                                                     freespace_discharge, peakload, loadLeft[h],
                                                                     Batt_SOC[h], dayHour[h])

        # Calculate total charge to battery throughout year for lifecycle analysis
        if P_batt[h] < 0:  # P_batt is negative when charging
            Batt_kWh_tot = Batt_kWh_tot - P_batt[h]

        # " calculate fuel usage "
        if P_gen[h] > 0:
            Fuel_kW, Fuel_kg = fuel_calcs(P_gen[h], peakload, timestep)
        else:
            Fuel_kg = 0

        # " Genset "
        Propane_kg = Propane_kg + Fuel_kg  # "Cumulative genset fuel consumption in kg Propane"

        # "increment loop variable"
        h += 1

    # "---------------------------------------------------------------------------------------"
    # "END OF LOOP"
    # "---------------------------------------------------------------------------------------"

    return Propane_kg, Batt_SOC, LoadkW, P_gen, P_PV, P_batt, P_dump, Limit_charge, Limit_discharge, Batt_kWh_tot


##=============================================================================================================================

# Plot Power Flows =============================================================
def PlotPowerFlows(P_PV, P_Batt, P_PG, P_dump, SOC, LoadkW, t1, t2, BattkW, Limit_discharge, Limit_charge):
    # This function creates plots of the power flows and battery state of charge for any specified
    # section of time throughout the year. The time period is specified by the starting time t1,
    # and the ending time t2.

    time = list(range(t1, t2))
    fig, ax = plt.subplots()
    ax.set_xlabel('time (h)')
    ax.set_ylabel('P (kW)')
    ax.plot(time, P_PV[t1:t2], color='green', linewidth=1)
    ax.plot(time, P_Batt[t1:t2], color='blue', linewidth=1)
    ax.plot(time, P_PG[t1:t2], color='red', linewidth=1)
    ax.plot(time, P_dump[t1:t2], color='cyan', linewidth=1)
    ax.plot(time, LoadkW[t1:t2], color='black', linewidth=1)
    ax.legend(['PV', 'Battery', 'PG', 'Dump', 'Load'])
    plt.xlim(t1, t2 - 1)
    plotname = "PowerFlows_JULY.png"
    plt.savefig(plotname, dpi=600)
    plt.show()

    SOC = np.array(SOC) / BattkW * 100
    Limit_discharge = np.ones(len(SOC)) * Limit_discharge
    Limit_charge = np.ones(len(SOC)) * -Limit_charge
    Zerosss = np.zeros(len(SOC))

    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].set_xlabel('time (h)')
    axarr[0].set_ylabel('P (kW)')
    axarr[0].plot(time, P_Batt[t1:t2], color='blue', linewidth=1)
    axarr[0].plot(time, Limit_discharge[t1:t2], color='black', linewidth=1, linestyle='dashed')
    axarr[0].plot(time, Limit_charge[t1:t2], color='black', linewidth=1, linestyle='dashed')
    axarr[0].plot(time, Zerosss[t1:t2], color='black', linewidth=1, linestyle=':')

    # ax.tick_params(axis='y',labelcolor = 'blue')

    axarr[1].plot(time, SOC[t1:t2], color='orange', linewidth=1)
    axarr[1].set_ylabel('SOC %')
    plt.ylim(-5, 105)
    # ax2.tick_params(axis='y',labelcolor = 'orange')
    # plt.legend(['Battery','SOC'])
    f.subplots_adjust(hspace=0)
    plt.xlim(t1, t2 - 1)
    plotname = "Battery_SOC_JULY.png"
    plt.savefig(plotname, dpi=600)
    plt.show()

##====================================================================
def get_8760(village_name):
   filtered_list = glob.glob(f'{village_name}*8760*.xlsx')
   for f in filtered_list:
       if village_name in f and '8760' in f:
           return f
   return None
    
##=============================================================================

# Plot Load =============================================================
def PlotLoad(LoadkW, t1, t2):
    # This function plots the load demand from the specified start time, t1,
    # to the end time, t2.

    time = list(range(t1, t2))
    fig, ax = plt.subplots()
    ax.set_xlabel('time (h)')
    ax.set_ylabel('P (kWh)')
    ax.plot(time, LoadkW[t1:t2], color='black', linewidth=1)
    plt.xlim(t1, t2 - 1)
    plotname = "LoadFlow_JAN.png"
    plt.savefig(plotname, dpi=600)
    plt.show()


##=============================================================================

## Tech_total function =========================================================================================================
def Tech_total(BattkWh_Parametric, PVkW_Parametric):
    # This function calls the previous functions, excluding the plotting function, to solve for the power flows for the year.
    sitename = SITE_NAME
   # Load Files
    load_file = get_8760(sitename)
    # print(load_file)
    load = pd.read_excel(loadfile, sheet_name='8760', usecols='B')
    # TODO: Here
    #FullYearEnergy = pd.read_excel('FullYearEnergy.xlsx', index_col=None, header=None) 
    FullYearEnergy = full_year_energy_calc(day_totals, modified8760, indices)
    #Test fullyear
    #print(FullYearEnergy)
    # TODO: Here
    TMY = pd.read_excel(sitename + '_TMY.xlsx')
    # TODO: Here
    Tech_Parameters = pd.read_excel(sitename + '_uGrid_Input.xlsx', sheet_name='Tech')
    # TODO: Here
    Solar_Parameters = pd.read_excel(sitename + '_uGrid_Input.xlsx', sheet_name='Solar')
    # TODO: Here
    #print(FullYearEnergy, TMY)

    # Input Calculations
    #print(load)
    peakload = max(load['kW']) * Tech_Parameters['peakload_buffer'][0]  # "maximum power output of the load curve [kW]"
    BattkWh = BattkWh_Parametric * peakload  # "[kWh]"
    loadkWh = sum(load['kW'])
    PVkW = PVkW_Parametric * peakload  # "[kW]"

    Propane_kg, Batt_SOC, LoadkW, P_gen, P_PV, P_batt, P_dump, Limit_charge, Limit_discharge, Batt_kWh_tot = operation(
        Tech_Parameters['Batt_Charge_Limit'][0], Tech_Parameters['smart'][0], PVkW, BattkWh, peakload, load,
        FullYearEnergy, TMY, Solar_Parameters, Tech_Parameters['trans_losses'][0])
    print(Batt_SOC, LoadkW)

    return Propane_kg, Batt_SOC, LoadkW, P_gen, P_PV, P_batt, P_dump, Limit_charge, Limit_discharge, BattkWh, Batt_kWh_tot, loadkWh, peakload


##=============================================================================================================


# " -----------------------------------------------------------------------------------------------------------"
# "SIMULTANEOUSLY SOLVED EQUATIONS" as a standalone program
# " -----------------------------------------------------------------------------------------------------------"
if __name__ == "__main__":
    # This python file can be run as standalone. In order to run as a standalone the BattkWh_Parametric and
    # PVkW_Parametric need to be specified. This is also where the plotting functions can be called, and t1 and t2 are specified.

    BattkWh_Parametric = 5.5
    PVkW_Parametric = 2.7

    Propane_kg, Batt_SOC, LoadkW, P_gen, P_PV, P_batt, P_dump, Limit_charge, Limit_discharge, BattkW, Batt_kWh_tot, loadkWh, peakload = Tech_total(
        BattkWh_Parametric, PVkW_Parametric)

    t1 = int(24 * 30 * 1)
    t2 = t1 + 25
    PlotPowerFlows(P_PV, P_batt, P_gen, P_dump, Batt_SOC, LoadkW, t1, t2, BattkW, Limit_discharge, Limit_charge)
