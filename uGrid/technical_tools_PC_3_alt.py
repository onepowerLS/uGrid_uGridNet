"""
This file contains all the required functions to run technical model standalone, to compare to the EES technical file. 

Created on Tue Jun 21 10:13:16 2016

@author: phylicia Cicilio
"""

from __future__ import division
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
#from numba import jit

## Battery Calculations ========================================================
def batt_calcs(timestep,BattkWh,T_amb,Batt_SOC_in,Limit_charge,Limit_discharge):
    
    #timestep = 1
    Self_discharge = timestep * 3600 * BattkWh * 3.41004573E-09 * np.exp(0.0693146968 * T_amb)   #"Effect of temperature on self discharge of the battery bank from 5% per month at 25C and doubling every 10C"
    Batt_SOC_in = Batt_SOC_in - Self_discharge
    SOC_frac = 0.711009126 + 0.0138667895 * T_amb - 0.0000933332591 * T_amb**2  #eta_{batt,temp}  #"Effect of temperature on the capacity of the battery bank from IEEE 485 Table 1"
    Batt_cap = BattkWh * SOC_frac
    high_trip = 0.95 * Batt_cap
    freespace_charge = high_trip - Batt_SOC_in
    if freespace_charge < 0:
        freespace_charge == 0
    low_trip = 0.05 * Batt_cap
    freespace_discharge = Batt_SOC_in - low_trip
    
    #account for charging and discharging limitrs
    Batt_discharge = min(Limit_discharge,freespace_discharge)
    Batt_charge = min(Limit_charge,freespace_charge)
    
    return Batt_discharge, Batt_charge,freespace_discharge #these are both positive

##=============================================================================
    
## Generation Control =========================================================
def GenControl(P_PV,L,Batt_discharge, Batt_charge,freespace_discharge,genPeak,LoadLeft,Batt_SOC,dayhour):
    
    if P_PV > 0:
        if P_PV - L < 0: #Not enough PV to power load, gen running, battery charging
            if freespace_discharge > LoadLeft and Batt_discharge > L-P_PV: #if in the morning (batteries can discharge to meet load)
                P_gen = 0
                P_batt = L - P_PV
                P_dump = 0
            else:
                P_gen = min(genPeak,(L-P_PV+Batt_charge))
                P_batt = -(P_gen + P_PV - L)
                P_dump = 0
        else: #More than enough PV for load, gen off, battery charging
            P_gen = 0
            P_batt = - min(P_PV - L, Batt_charge)
            P_dump = P_PV + P_batt - L
    else:
        if freespace_discharge > LoadLeft and Batt_discharge > L: #enough charge to run through night and meet load demand, PV and gen off
            P_gen = 0
            P_batt = np.copy(L)
            P_dump = 0
        else: # not enough charge, PV off, gen on, battery charging
            P_gen = min(genPeak,(L+Batt_charge))
            P_batt = -(P_gen-L)
            P_dump = 0
    
    Batt_SOC = Batt_SOC - P_batt
    
    return P_batt, P_gen, Batt_SOC, P_dump
#==============================================================================


## fuel_calcs FUNCTION =====================================================================================================
#@jit(nopython=True) # Set "nopython" mode for best performance
def fuel_calcs(genload,peakload,timestep):
    #"generator fuel consumption calculations based on efficiency as a function of power output fraction of nameplate rating"
   
    partload = genload / peakload
    Eta_genset = -0.00430876206 + 0.372448046*partload - 0.174532718*partload**2   #"Derived from Onan 25KY model"
 
    if Eta_genset < 0.02:
        Eta_genset=0.02  #"prevents erroneous results at extreme partload operation"

    Fuel_kW = genload/Eta_genset  #"[kW]"
 
	#"Fuel consumption"
    Fuel_kJ=Fuel_kW*timestep*3600   #"[J]"
    Fuel_kg=Fuel_kJ/50340   #"[converts J to kg propane]"
    
    return Fuel_kW,Fuel_kg
##=============================================================================================================================    


## operation FUNCTION =====================================================================================================
def operation(Batt_Charge_Limit,low_trip_perc,high_trip_perc,lowlightcutoff,Pmax_Tco, NOCT, smart, PVkW, BattkWh, peakload, LoadKW_MAK, FullYearEnergy, MSU_TMY,Solar_Parameters,trans_losses):
    from solar_calcs_PC_3 import SolarTotal
    
    # This code can and should be cleaned up *****
    
    # Month$ was replaced with Month_

    timestep=1 #"hours - initialize timestep to 10 minutes"
 
    #Find length of Load Dataset
    hmax = len(LoadKW_MAK)
    Hour = np.arange(hmax)
      
    #initialize arrays
    P_PV = np.zeros(hmax)  #"kW electrical"
    P_batt = np.zeros(hmax)
    P_gen = np.zeros(hmax)
    P_dump = np.zeros(hmax)
    Batt_SOC = np.zeros(hmax+1)
    dayHour = np.zeros(hmax)
    LoadkW = np.zeros(hmax)
    loadLeft = np.zeros(hmax)
  
    #"Initialize variables"
    Propane=0
    Limit_discharge = BattkWh/Batt_Charge_Limit
    Limit_charge = BattkWh/Batt_Charge_Limit
    
    #" If battery bank is included in generation equipment "
    if BattkWh > 0: 
            Batt_SOC[0]=BattkWh/4 #" Initialize the Batt Bank to half full"
    else:
    	Batt_SOC[0]=0 #"There is no Batt Bank"
    
    Batt_kWh_tot = 0
 
    #"derive factor for latitude tilted PV panel based on I_tilt to DNI ratio from NASA SSE dataset - used for fixed tilt"
    #CALL I_Tilt(Lat, Long, Month:lat_factor)
    #lat_factor=1
 
    #"---------------------------------------------------------------------------------------"
    #"WRAPPER LOOP"
    #"---------------------------------------------------------------------------------------"

    h=0 #"initialize loop variable (timestep counter)"
    while h < hmax-16:
 
    	#"============================="
        #" Analyze external conditions "
        #"============================="
 
        #"Establish the time of day in hours"
        if Hour[h] > 24:  #"factor the timestep down to get the time of day"
            factor = math.floor(Hour[h]/24)
            dayHour[h]=Hour[h]-factor*24
        else: 
            dayHour[h] = np.copy(Hour[h])  #"timestep is the time of day during the first day"
 
        if dayHour[h] == 24:
            dayHour[h]=0
 
        if PVkW > 0:
            #" Assess solar availability "
            hrang,declin,theta,Gt,P_PV[h],T_amb = SolarTotal(MSU_TMY,Solar_Parameters['year'][0],Hour[h],Solar_Parameters['longitude'][0],Solar_Parameters['latitude'][0],Solar_Parameters['timezone'][0],Solar_Parameters['slope'][0],Solar_Parameters['azimuth'][0],Solar_Parameters['pg'][0],PVkW,Solar_Parameters['fpv'][0],Solar_Parameters['alpha_p'][0],Solar_Parameters['eff_mpp'][0],Solar_Parameters['f_inv'][0])
        else:
            T_amb=-9999
            P_PV[h]=0
	
 
        #"LOAD (kW) for this timestep"
        #"calculate the row value in the Nkautest simulated load dataset (based on Hobhouse NRS) corresponding to the simulation timestep"
        loadcounter=Hour[0]/timestep + h  
        LoadkW[h] = LoadKW_MAK.iloc[int(loadcounter),0]*(1+trans_losses) #"choose the reference load dataset" #calls from lookuptable. Need to load excel sheet of this data
        #LOOKUP['Table Name', Row, Column]=Value, Set the value of the cell at the specified row andcolumn of the specified Lookup table to the given value. Column must either be an integer, referringto the column number or a string constant or string variable that provides the name of the column.
  	
 
        #"*******************************************************************************************************"
        #"Smart charging strategy - the generator will only continue charging the battery bank if the SOC of the bank "
        #"is less than the SOC which would permit inverter operation until dawn (when the PV can ideally be prioritized "
        #"to recharge the bank at a lower operating cost than genset charging)"
        #"*******************************************************************************************************"
 
        if smart == 1:
            #" estimate load remaining until dawn for nighttimes (set to zero for daytimes) "
            #if dayHour[h] > 17 or dayHour[h] < 7: #"determine the amount of load in kWh remaining from premidnight until dawn"
            #    loadLeft[h] = FullYearEnergy.iloc[int(loadcounter-1),0]*0.75/5  #"Modify Nkau for number of households in community of interest" # I added *1.2 to have a little more overshoot
                #the FullYearEnergy provides the amount of forecasted nighttime load at each hour through the night (most loadleft at beginning of night, least load left at end of night)
            #else:
            #    loadLeft[h]=0 #"daytime load prediction not part of algorithm"
                
            #Try considering max LoadLeft for the next 12 hours
            loadLeft[h] = max(list(FullYearEnergy[h:h+10][0]))*0.75/5*1.5
        else:
            loadLeft[h] = 10000+2*BattkWh #"number much higher than can possibly be stored in battery bank"
            #" forces genset to stay on (not smart) "
 
        Batt_discharge, Batt_charge, freespace_discharge = batt_calcs(timestep,BattkWh,T_amb,Batt_SOC[h],Limit_charge,Limit_discharge)
        P_batt[h], P_gen[h], Batt_SOC[h+1], P_dump[h] = GenControl(P_PV[h],LoadkW[h],Batt_discharge, Batt_charge,freespace_discharge,peakload,loadLeft[h],Batt_SOC[h],dayHour[h])
        
        #Calculate total charge to battery throughout year for lifecycle analysis
        if P_batt[h] < 0: #P_batt is negative when charging
            Batt_kWh_tot = Batt_kWh_tot - P_batt[h]
        
        #" calculate fuel usage "
        if P_gen[h] > 0:
            Fuel_kW,Fuel_kg = fuel_calcs(P_gen[h],peakload,timestep)
        else:
            Fuel_kg = 0
 
        #" Genset "
        Propane = Propane+Fuel_kg    #"Cumulative genset fuel consumption in kg Propane"					
 
        #"increment loop variable"
        h += 1

 
    #"---------------------------------------------------------------------------------------"
    #"END OF WRAPPER LOOP"
    #"---------------------------------------------------------------------------------------"
    
    return Propane, Batt_SOC, LoadkW, P_gen, P_PV, P_batt, P_dump, Limit_charge, Limit_discharge, Batt_kWh_tot

##============================================================================================================================= 

#Plot Power Flows =============================================================
def PlotPowerFlows(P_PV,P_Batt,P_PG,P_dump,SOC,LoadkW,t1,t2,BattkW,Limit_discharge,Limit_charge):
    
    time = list(range(t1,t2))
    fig, ax = plt.subplots()
    ax.set_xlabel('time (h)')
    ax.set_ylabel('P (kWh)')
    ax.plot(time, P_PV[t1:t2],color='green',linewidth=1)
    ax.plot(time, P_Batt[t1:t2],color='blue',linewidth=1)
    ax.plot(time, P_PG[t1:t2],color='red',linewidth=1)
    ax.plot(time, P_dump[t1:t2],color='cyan',linewidth=1)
    ax.plot(time, LoadkW[t1:t2],color='black',linewidth=1)
    ax.legend(['PV','Battery','PG','Dump','Load'])
    plt.xlim(t1,t2-1)
    plotname = "PowerFlows.png"
    plt.savefig(plotname, dpi=600)
    plt.show()
    
    SOC = np.array(SOC)/BattkW*100
    Limit_discharge = np.ones(len(SOC))*Limit_discharge
    Limit_charge = np.ones(len(SOC))*-Limit_charge
    Zerosss = np.zeros(len(SOC))
    
    f, axarr = plt.subplots(2,sharex=True)
    axarr[0].set_xlabel('time (h)')
    axarr[0].set_ylabel('P (kWh)')
    axarr[0].plot(time, P_Batt[t1:t2],color='blue',linewidth=1)
    axarr[0].plot(time, Limit_discharge[t1:t2],color='black',linewidth=1,linestyle='dashed')
    axarr[0].plot(time, Limit_charge[t1:t2],color='black',linewidth=1,linestyle='dashed')
    axarr[0].plot(time, Zerosss[t1:t2],color='black',linewidth=1,linestyle=':')
    
    #ax.tick_params(axis='y',labelcolor = 'blue')
    
    axarr[1].plot(time, SOC[t1:t2],color='orange',linewidth=1)
    axarr[1].set_ylabel('SOC %')
    plt.ylim(-5,105)
    #ax2.tick_params(axis='y',labelcolor = 'orange')
    #plt.legend(['Battery','SOC'])
    f.subplots_adjust(hspace=0)
    plt.xlim(t1,t2-1)
    plotname = "Battery_SOC.png"
    plt.savefig(plotname, dpi=600)
    plt.show()
##=============================================================================

## Tech_total function =========================================================================================================
def Tech_total(BattkWh_Parametric,PVkW_Parametric):

    #Load excel files containing LoadKW_MAK, FullYearEnergy, final
    LoadKW_MAK = pd.read_excel('LoadKW_MAK.xlsx',index_col=None, header=None)
    FullYearEnergy = pd.read_excel('FullYearEnergy.xlsx',index_col=None, header=None)
    MSU_TMY = pd.read_excel('MSU_TMY.xlsx')
    Tech_Parameters = pd.read_excel('uGrid_Input.xlsx', sheet_name = 'Tech')
    Solar_Parameters = pd.read_excel('uGrid_Input.xlsx', sheet_name = 'Solar')     
    
    #Input Calculations
    peakload=max(LoadKW_MAK[0])*Tech_Parameters['peakload_buffer'][0] #"maximum power output of the load curve [kW]"
    BattkWh=BattkWh_Parametric*peakload  #"[kWh]"
    loadkWh = sum(LoadKW_MAK[0])
    PVkW=PVkW_Parametric*peakload  #"[kW]"
    
    Propane, Batt_SOC, LoadkW, P_gen, P_PV, P_batt, P_dump,Limit_charge, Limit_discharge, Batt_kWh_tot = operation(Tech_Parameters['Batt_Charge_Limit'][0],Tech_Parameters['low_trip_perc'][0],Tech_Parameters['high_trip_perc'][0],Tech_Parameters['lowlightcutoff'][0],Tech_Parameters['Pmax_Tco'][0], Tech_Parameters['NOCT'][0], Tech_Parameters['smart'][0], PVkW, BattkWh, peakload, LoadKW_MAK, FullYearEnergy, MSU_TMY,Solar_Parameters,Tech_Parameters['trans_losses'][0]) 
    
    return Propane, Batt_SOC, LoadkW, P_gen, P_PV, P_batt, P_dump, Limit_charge, Limit_discharge,BattkWh,Batt_kWh_tot,loadkWh,peakload
##=============================================================================================================



#" -----------------------------------------------------------------------------------------------------------"
#"SIMULTANEOUSLY SOLVED EQUATIONS" as a standalone program
#" -----------------------------------------------------------------------------------------------------------"
if __name__ == "__main__":

    BattkWh_Parametric=6
    PVkW_Parametric=2
    
    Propane, Batt_SOC, LoadkW, P_gen, P_PV, P_batt, P_dump,Limit_charge, Limit_discharge, BattkW, Batt_kWh_tot,loadkWh,peakload = Tech_total(BattkWh_Parametric,PVkW_Parametric)
    
    t1=0
    t2=25
    PlotPowerFlows(P_PV,P_batt,P_gen,P_dump,Batt_SOC,LoadkW,t1,t2,BattkW,Limit_discharge,Limit_charge)