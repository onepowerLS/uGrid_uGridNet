"""
This file contains all the required functions to run technical model standalone, to compare to the EES technical file. 

Created on Tue Jun 21 10:13:16 2016

@author: phylicia Cicilio
"""

from __future__ import division
import numpy as np
import pandas as pd
import math
from numba import jit


## Batt_Bal FUNCTION =====================================================================================================
#@jit(nopython=True) # Set "nopython" mode for best performance
def batt_bal(T_amb, Limit, PV_Batt_Charge_Power,Inv_Batt_dis_P,Gen_Batt_Charge_Power,Batt_SOC_in,BattkWh,timestep): 
#REMOVE ORC
    
    Positive_Charge = 0  #"initialize the battery duty variable"
    #"ACCOUNT FOR TEMPERATURE AND DEPTH OF DISHARGE EFFECTS ON BATT BANK"
 
    #Don't like all these hard coded numbers
    Self_discharge = timestep * 3600 * BattkWh * 3.41004573E-09 * np.exp(0.0693146968 * T_amb)   #"Effect of temperature on self discharge of the battery bank from 5% per month at 25C and doubling every 10C"
    Batt_SOC_in = Batt_SOC_in - Self_discharge
    SOC_frac = 0.711009126 + 0.0138667895 * T_amb - 0.0000933332591 * T_amb**2   #"Effect of temperature on the capacity of the battery bank from IEEE 485 Table 1"
    high_trip = 0.95 * BattkWh * SOC_frac
 
    #"ENERGY BALANCE ON BATT BANK"
    #"Possible charge/discharge mechanisms accounted for"
    Charge = PV_Batt_Charge_Power + Inv_Batt_dis_P + Gen_Batt_Charge_Power #charge of the battery (Inv is negative, power leaving battery)
    #print Inv_Batt_dis_P	
    space_avail = BattkWh*SOC_frac - Batt_SOC_in 	#"kWh"     "remaining capacity to fill"
    
    #" if there are no batteries, all energy gets dumped "
    if BattkWh == 0:
        Charge = 0
        dumpload = PV_Batt_Charge_Power + Gen_Batt_Charge_Power
        Batt_SOC_out = 0
        if Inv_Batt_dis_P < 0:
		    print "Warning: System has no Battery but Inverter is operational -> ERROR!!!"
    else:
	    #"Batteries cannot be charged beyond full capacity, extra goes to a dump"
        Charge_e = Charge * timestep #"kWh"  "Charge energy"
        Limit_e = Limit * timestep  #"kWh"    "Acceptable charge energy in this timestep based on the charge rate limit"
        if Charge_e > space_avail:  # "Charge energy exceeds available capacity in the batt bank"
            if space_avail > Limit_e:
                dumpload = Charge - Limit
                Charge = np.copy(Limit)
            else:    #"Available capacity in the battery bank exceeds the energy that can be delivered in this time step under the charge limit"
                dumpload = Charge - space_avail / timestep
                Charge = space_avail / timestep
        else:    #"amount for charging is < amount needed to fill batteries"
            if Charge > Limit:
                #"Batt bank charging is limited to a fraction of its rated capacity"
               dumpload = Charge - Limit
               Charge = np.copy(Limit)
            else:
                #"all charging energy can be used"
                #"Charge = Charge"
                dumpload = 0
 
	   	#"State of charge in the next timestep is a function of source logic above"
 	   	#"90% one way efficiency at charge/discharge"
        if Charge > 0:
            Batt_SOC_out = Batt_SOC_in + Charge * timestep * 0.9
			#"Track the input to the battery bank for life cycle calculation"
            Positive_Charge = np.copy(Charge)
        if Charge <= 0:
            Batt_SOC_out = Batt_SOC_in + Charge * timestep / 0.9
            Positive_Charge = 0
        
    return Positive_Charge,Charge,Batt_SOC_out,dumpload,high_trip
##=============================================================================================================================
    

## Storage_Levels FUNCTION =====================================================================================================
@jit(nopython=True) # Set "nopython" mode for best performance
def storage_levels(Batt_SOC,low_trip,high_trip,BattkWh,loadLeft):
#Compare the SOC of the battery with the battery limits to determine if battery 
#will be charging, supplying, or doing nothing. 
    
    #" Battery SOC cases "
    if BattkWh == 0:
        battcase = 1 #"if no batteries, always responds as Low Batt"
        #battcase 1: No charge in battery
    else:
        if Batt_SOC > high_trip:
            battcase = 4
            #battcase 4: Battery is over high limit, do not charge but can discharge
        else:
            if Batt_SOC < low_trip:
                battcase = 1
                #battcase 1: Battery is below low limit, do not discharge but can charge
            else:
                if (Batt_SOC-low_trip) > loadLeft:
                    battcase = 3
                    #battcase 3: Charge is battery is enough to supply load left
                    #without going below battery's low limit
                else:
                    battcase = 2
                    #battcase 2: Battery is capable of charging or discharging
 
    return battcase
##=============================================================================================================================


## GetState FUNCTION =====================================================================================================
@jit(nopython=True) # Set "nopython" mode for best performance
def getstate(PV_Pow,loadkW,battcase,BattkWh, peakload):

    #This can be cleaned up
    
    diff = PV_Pow - loadkW    #"difference between PV output and load"

    #"PV OUTPUT ON/OFF"
    #" recall low-light cutoff and PV_kW=0 checks in Solar Ambient"

    if PV_Pow > 0:    #PV is on
        if battcase == 1 and diff < 0: #Genset is on
            current_state = 5
        else:
            current_state = 2 #genset is off
    else: #PV on
        if battcase == 4 or battcase == 3: 
            current_state = 1 #genset off
        else:
            current_state = 4 #genset on
            
  
    return current_state
##=============================================================================================================================    


## setVars FUNCTION =====================================================================================================
def setvars(current_state,I_b,lowlightcutoff,PV_Avail_Pow,timestep,T_amb,Genset_peak,loadkW,BattkWh,Limit):
    #" Set system variable values based on current state number "
 
    #"check if PV power is greater than or less than load demand"							
    diff = PV_Avail_Pow - loadkW	#"difference between potential PV output and load"
    #print diff
 
    #"NOTES:"
    #"If Inverter makes up the difference from the Battery Bank, the (dis)charge is negative IF the generator is off"
 
    if current_state == 1:
        #" no generation equipment is on"
        #" PV, ORC, Genset = 0 0 0 "
        PV_Pow = 0
        PV_Batt_Charge_Power = 0  #"there is no power from the PV"
 
        #"load is powered from the inverter"
        Inv_Batt_Dis_P = np.copy(-loadkW)
 
        genload = 0
        Gen_Batt_Charge_Power = 0
        
    if current_state == 2:
        #" PV, ORC, Genset = 1 0 0 "
        PV_Pow = np.copy(PV_Avail_Pow)
        
        if diff > 0: #"if the difference is positive there is power available"
            PV_Batt_Charge_Power = np.copy(diff)   #"PV charging batteries"
            Inv_Batt_Dis_P = 0	    #"PV is powering entire load"
        else: #"the difference is negative, the load is more than the PV can supply"
            PV_Batt_Charge_Power=0  #"there is no power surplus from the PV"
        #    "load is powered from the inverter -- unless batteries are low (condition checked when state is set)"
            Inv_Batt_Dis_P = np.copy(diff)
 
        genload = 0
        Gen_Batt_Charge_Power = 0
 
    if current_state == 4:
        #" PV, ORC, Genset = 0 0 1 "
        PV_Pow = 0
        PV_Batt_Charge_Power = 0  #"there is no power from the PV"
        Inv_Batt_Dis_P = 0  #"the inverter is not supplying any loads if the genset is on"
 
        genload = np.copy(loadkW)  #"The Genset supplies entire load"
 
        #"Determine whether and how much the genset is charging the battery bank based on presence of a battery bank, residual genset capacity, and battery charge rate limit"
        if BattkWh > 0:  #"IF there is a battery and"
            if genload < Genset_peak:
                #"if loads fall below its peak output the generator will charge the battery bank up to the full charge rate"	
                Difference = Genset_peak - genload   #"residual genset capacity"
                if Difference > Limit:
                    #"if capacity is higher than the charge rate limit, the genset charges at the limit"
                    Gen_Batt_Charge_Power = np.copy(Limit)
                else:
                    #"the genset charges up to its full residual capacity after supplying the load"
                    Gen_Batt_Charge_Power = np.copy(Difference)  
 
                genload = genload + Gen_Batt_Charge_Power
            else:
                #" supplying peak load "
                Gen_Batt_Charge_Power = 0	
        else:
            #there is no battery, therefore it can't be charged
            Gen_Batt_Charge_Power=0


    if current_state == 5:
         #" PV, ORC, Genset = 1 0 1 "
         #" being in this state means battery is low AND PV inadequate to power load "
 
        PV_Pow = np.copy(PV_Avail_Pow)
        PV_Batt_Charge_Power=0  #"there is no power surplus from the PV"
        Inv_Batt_Dis_P = 0  #"the inverter is not supplying any loads if the genset is on"
        genload = loadkW-PV_Pow  #"The Genset makes up the difference between the load and the PV output"
 
        #"Determine whether and how much the genset is charging the battery bank based on presence of a battery bank, residual genset capacity, and battery charge rate limit"
        if BattkWh > 0:  #"IF there is a battery and"
            if genload < Genset_peak:
                #"if loads fall below its peak output the generator will charge the battery bank up to the full charge rate"	
                Difference = Genset_peak-genload   #"residual genset capacity"
                if Difference > Limit:
                    #"if capacity is higher than the charge rate limit, the genset charges at the limit"
                    Gen_Batt_Charge_Power = np.copy(Limit)
                else:
                    #"the genset charges up to its full residual capacity after supplying the load"
                    Gen_Batt_Charge_Power = np.copy(Difference)  
                
                genload = genload + Gen_Batt_Charge_Power
            else:
                #" supplying peak load "
                Gen_Batt_Charge_Power = 0	
        else:
            Gen_Batt_Charge_Power=0
 
 
 
    return PV_Pow, PV_Batt_Charge_Power,Inv_Batt_Dis_P,genload,Gen_Batt_Charge_Power

##============================================================================================================================= 


## fuel_calcs FUNCTION =====================================================================================================
@jit(nopython=True) # Set "nopython" mode for best performance
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


## solar_ambient FUNCTION =====================================================================================================
def solar_ambient(th_Hour,NOCT,lat_factor,Pmax_Tco,PVkW,lowlightcutoff,MSU_TMY):
    #"SOLAR - AMBIENT"
    #"determine ambient conditions and perform the calculations related to irradiance and temperature"
    #"Set beam radiation equal to the irradiance distribution calculated in the solar module"
    Hour = MSU_TMY.loc[:,'Hour']  #collects entire column
    Tamb = MSU_TMY.loc[:,'Tamb'] 
    Itrack = MSU_TMY.loc[:,'Itrack'] #in the TMY spreadsheet DNI, GHI, and itrack are all the same

    #"derive T_amb from TMY dataset"
    T_amb = np.interp(th_Hour, Hour, Tamb) #('MSU_TMY','Tamb','Hour',Hour=th_Hour)	 
	
    if PVkW > 0:
        #"derive I_b from DNI dataset or modified dataset for tracker"
        #I_b=interpolate('MSU_TMY','Itrack','Hour',Hour=th_Hour)
        I_b = np.interp(th_Hour, Hour, Itrack)  
    else:
        I_b=-999
 
    if I_b > lowlightcutoff: 
        if PVkW > 0:
            #"Determine the effect of temperature on the PV cell efficiency"
            #"T_Cell in relation to NOCT, irradiance and T_amb from Handbook of Photovoltaic Science and Engineering, Hegedus/Luque 2011  pg 801
            #modified DNI to account for latitutude tilt"
            T_cell=T_amb+(NOCT-20)*I_b/800   #"[C]"  "*lat_factor if fixed tilt"
            P_norm=((100+(T_cell-25)*Pmax_Tco)/100)*1.2  #"add 20% for tracking"
            #"PV Calculations"
            PV_Avail_Pow=PVkW*P_norm*I_b/1000   #"scale with I_b and adjust for lat tilt to DNI ratio, and include P_norm to account for temperature coefficient"
        else:
            PV_Avail_Pow = 0
    else:
        PV_Avail_Pow = 0


    return I_b,T_amb,PV_Avail_Pow
##=============================================================================================================================  


## operation FUNCTION =====================================================================================================
def operation(Batt_Charge_Limit,low_trip_perc,high_trip_perc,lowlightcutoff,Pmax_Tco, NOCT, smart, PVkW, BattkWh, peakload, LoadKW_MAK, FullYearEnergy, MSU_TMY):
# Month$ was replaced with Month_

    timestep=1 #"hours - initialize timestep to 10 minutes"
 
    #Find length of Load Dataset
    hmax = len(LoadKW_MAK)
    Hour = np.arange(hmax)
      
    #initialize arrays
    State = np.zeros(hmax)
    beam = np.zeros(hmax)  #"to track irradiance"
    T_ambient = np.zeros(hmax)  #"to track ambient temperature"
    PV_Power = np.zeros(hmax)  #"kW electrical"
    PV_Batt_Charge_Power = np.zeros(hmax)
    Inv_Batt_Dis_P = np.zeros(hmax)
    Charge = np.zeros(hmax)
    dumpload = np.zeros(hmax)
    Batt_SOC = np.zeros(hmax)
    Batt_frac = np.zeros(hmax)
    genload = np.zeros(hmax)
    Gen_Batt_Charge_Power = np.zeros(hmax)
    Genset_fuel = np.zeros(hmax)
    Fuel_kW = np.zeros(hmax)
    dayHour = np.zeros(hmax)
    LoadkW = np.zeros(hmax)
    loadLeft = np.zeros(hmax)
  
    #"Initialize variables"
    DNI=0
    Propane=0
    Limit = BattkWh/Batt_Charge_Limit	#"kW"  "Battery can only be charged at rate up to arbitrary 1/5 of its full capacity rating"
    high_trip = high_trip_perc*BattkWh	  #" kWh "
    low_trip = low_trip_perc*BattkWh	 # " kWh "
    I_b=0
    genload[0] = 0
    Gen_Batt_Charge_Power[0] = 0
    PV_Batt_Charge_Power[0] = 0
    Inv_Batt_Dis_P[0] =0
    Batt_kWh_tot=0 
 
    #" If battery bank is included in generation equipment "
    if BattkWh > 0: 
            Batt_SOC[0]=BattkWh/2 #" Initialize the Batt Bank to half full"
    else:
    	Batt_SOC[0]=0 #"There is no Batt Bank"
 
    #"derive factor for latitude tilted PV panel based on I_tilt to DNI ratio from NASA SSE dataset - used for fixed tilt"
    #CALL I_Tilt(Lat, Long, Month:lat_factor)
    lat_factor=1
 
    #"---------------------------------------------------------------------------------------"
    #"WRAPPER LOOP"
    #"---------------------------------------------------------------------------------------"

    h=0 #"initialize loop variable (timestep counter)"
    while h < hmax:
 
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
            I_b,T_amb,PV_Avail_Pow = solar_ambient(Hour[h],NOCT,lat_factor,Pmax_Tco,PVkW,lowlightcutoff, MSU_TMY)
        else:
            T_amb=-9999
            PV_Avail_Pow=0
            I_b=-9999
	
 
        #"LOAD (kW) for this timestep"
        #"calculate the row value in the Nkautest simulated load dataset (based on Hobhouse NRS) corresponding to the simulation timestep"
        loadcounter=Hour[0]/timestep + h  
        LoadkW[h] = LoadKW_MAK.iloc[int(loadcounter),0] #"choose the reference load dataset" #calls from lookuptable. Need to load excel sheet of this data
        #LOOKUP['Table Name', Row, Column]=Value, Set the value of the cell at the specified row andcolumn of the specified Lookup table to the given value. Column must either be an integer, referringto the column number or a string constant or string variable that provides the name of the column.
  	
 
        #"*******************************************************************************************************"
        #"Smart charging strategy - the generator will only continue charging the battery bank if the SOC of the bank "
        #"is less than the SOC which would permit inverter operation until dawn (when the PV can ideally be prioritized "
        #"to recharge the bank at a lower operating cost than genset charging)"
        #"*******************************************************************************************************"
 
        if smart == 1:
            #" estimate load remaining until dawn for nighttimes (set to zero for daytimes) "
            if dayHour[h] > 17 or dayHour[h] < 7: #"determine the amount of load in kWh remaining from premidnight until dawn"
                loadLeft[h] = FullYearEnergy.iloc[int(loadcounter-1),0]*0.75/5   #"Modify Nkau for number of households in community of interest"
            else:
                loadLeft[h]=0 #"daytime load prediction not part of algorithm"
        else:
            loadLeft[h] = 10000+2*BattkWh #"number much higher than can possibly be stored in battery bank"
            #" forces genset to stay on (not smart) "
 
        #" how full are thermal, chemical storage? "
        battcase = storage_levels(Batt_SOC[h],low_trip,high_trip,BattkWh,loadLeft[h])
        
        #"check for error conditions" THIS WAS ALL COMMENTED OUT
        #If ( (CSP_A = 0) AND (TES_SOC[h] > lowT_trip) ) Then
            #" THROW ERROR "
        #If ( (BattkWh = 0) AND (Batt_SOC[h] > low_trip) ) Then
            #" THROW ERROR "
        #If ( (I_b < lowlightcutoff) AND (PV_Power[h] > LoadkW[h]) ) Then
            #" THROW ERROR "
        #If ( (PV_kW = 0) AND (PV_Power[h] > LoadkW) ) Then
            #" THROW ERROR "
 
        #" determine state "
        current_state = getstate(PV_Avail_Pow,LoadkW[h],battcase,BattkWh,peakload)
 
        #" Set system variables "
        PV_Pow, PV_BCP,Inv_BDP,genL,Gen_BCP = setvars(current_state,I_b,lowlightcutoff,PV_Avail_Pow,timestep,T_amb,peakload,LoadkW[h],BattkWh,Limit)
 
        #" Balance energy storage in batteries "
        Positive_Charge,Charge_,Batt_SOC_out,dump,high_trip = batt_bal(T_amb, Limit, PV_BCP,Inv_BDP,Gen_BCP,Batt_SOC[h],BattkWh,timestep) 
 
        #" calculate fuel usage "
        if genL > 0:
            FkW,Fuel_kg = fuel_calcs(genL,peakload,timestep)
        else:
            FkW = 0
            Fuel_kg = 0

    	#"============================="
        #" Update cumulative variables "
        #"============================="
 
    	#"State variables"
        State[h]=current_state
 
        #" Env parameters "
        if h<hmax-1:
            Hour[h+1] = Hour[h]+timestep
        beam[h] = np.copy(I_b)  #"to track irradiance"
        DNI = DNI+(beam[h]*timestep)/1000
	
        T_ambient[h] = np.copy(T_amb)  #"to track ambient temperature"
 
        #" Solar "
        PV_Power[h] = np.copy(PV_Pow)  #"kW electrical"
        PV_Batt_Charge_Power[h] = np.copy(PV_BCP)
        Inv_Batt_Dis_P[h] = np.copy(Inv_BDP)
 
        #" batteries "
        Charge[h] = np.copy(Charge_)
        #print Positive_Charge
        Batt_kWh_tot=Batt_kWh_tot+Positive_Charge*timestep  #"amout of kWh flowing through the battery"
        dumpload[h] = np.copy(dump)
        Batt_SOC[h] = np.copy(Batt_SOC_out)
        if BattkWh > 0:
    		Batt_frac[h]=Batt_SOC[h]/BattkWh
        else:
            Batt_frac[h]=0
        
        if h < hmax-1:
            Batt_SOC[h+1] = np.copy(Batt_SOC[h])
 
        #" Genset "
        genload[h] = np.copy(genL)
        Gen_Batt_Charge_Power[h] = np.copy(Gen_BCP)
        Genset_fuel[h] = np.copy(Fuel_kg)
        Fuel_kW[h] = np.copy(FkW)
        Propane = Propane+Genset_fuel[h]    #"Cumulative genset fuel consumption in kg Propane"					
 
        #"increment loop variable"
        h += 1

 
    #"---------------------------------------------------------------------------------------"
    #"END OF WRAPPER LOOP"
    #"---------------------------------------------------------------------------------------"
    
    return Propane, DNI, Batt_kWh_tot, Batt_SOC, Charge, State, LoadkW, genload, Inv_Batt_Dis_P, PV_Power, PV_Batt_Charge_Power, dumpload, Batt_frac, Gen_Batt_Charge_Power, Genset_fuel, Fuel_kW

##============================================================================================================================= 

## Tech_total function =========================================================================================================
def Tech_total(BattkWh_Parametric,PVkW_Parametric):

    #Load excel files containing LoadKW_MAK, FullYearEnergy, final
    LoadKW_MAK = pd.read_excel('LoadKW_MAK.xlsx',index_col=None, header=None)
    FullYearEnergy = pd.read_excel('FullYearEnergy.xlsx',index_col=None, header=None)
    MSU_TMY = pd.read_excel('MSU_TMY.xlsx')
    Tech_Parameters = pd.read_excel('uGrid_Input.xlsx', sheet_name = 'Tech')    
    
    #Input Calculations
    peakload=max(LoadKW_MAK[0])*Tech_Parameters['peakload_buffer'][0] #"maximum power output of the load curve [kW]"
    loadkWh = sum(LoadKW_MAK[0])
    BattkWh=BattkWh_Parametric*peakload  #"[kWh]"
    PVkW=PVkW_Parametric*peakload  #"[kW]"
    
    Propane, DNI, Batt_kWh_tot, Batt_SOC, Charge, State,LoadkW, genLoad, Inv_Batt_Dis_P, PV_Power, PV_Batt_Charge_Power, dumpload, Batt_frac, Gen_Batt_Charge_Power, Genset_fuel, Fuel_kW = operation(Tech_Parameters['Batt_Charge_Limit'][0],Tech_Parameters['low_trip_perc'][0],Tech_Parameters['high_trip_perc'][0],Tech_Parameters['lowlightcutoff'][0],Tech_Parameters['Pmax_Tco'][0], Tech_Parameters['NOCT'][0], Tech_Parameters['smart'][0], PVkW, BattkWh, peakload, LoadKW_MAK, FullYearEnergy, MSU_TMY) 
    
    return Propane, DNI, Batt_kWh_tot, Batt_SOC, Charge, State, LoadkW, genLoad, Inv_Batt_Dis_P, PV_Power, PV_Batt_Charge_Power, dumpload, Batt_frac, Gen_Batt_Charge_Power, Genset_fuel, Fuel_kW,peakload,loadkWh
##=============================================================================================================



#" -----------------------------------------------------------------------------------------------------------"
#"SIMULTANEOUSLY SOLVED EQUATIONS" as a standalone program
#" -----------------------------------------------------------------------------------------------------------"
if __name__ == "__main__":

    BattkWh_Parametric=7
    PVkW_Parametric=2.36
    
    Propane, DNI, Batt_kWh_tot, Batt_SOC, Charge, State, LoadkW, genLoad, Inv_Batt_Dis_P, PV_Power, PV_Batt_Charge_Power, dumpload, Batt_frac, Gen_Batt_Charge_Power, Genset_fuel, Fuel_kW,peakload,loadkWh = Tech_total(BattkWh_Parametric,PVkW_Parametric)