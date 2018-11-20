"""
This file contains all the required functions to run technical model standalone, to compare to the EES technical file. 

Created on Tue Jun 21 10:13:16 2016

@author: phylicia Cicilio
"""

from __future__ import division
import numpy as np
import pandas as pd
import math


## Batt_Bal FUNCTION =====================================================================================================
def batt_bal(T_amb, Limit, PV_Batt_Charge_Power,Inv_Batt_dis_P,Gen_Batt_Charge_Power,ORC,Batt_SOC_in,BattkWh,timestep): 
#REMOVE ORC
    
    Positive_Charge = 0  #"initialize the battery duty variable"
    #"ACCOUNT FOR TEMPERATURE AND DEPTH OF DISHARGE EFFECTS ON BATT BANK"
 
    Self_discharge = timestep * 3600 * BattkWh * 3.41004573E-09 * np.exp(0.0693146968 * T_amb)   #"Effect of temperature on self discharge of the battery bank from 5% per month at 25C and doubling every 10C"
    Batt_SOC_in = Batt_SOC_in - Self_discharge
    SOC_frac = 0.711009126 + 0.0138667895 * T_amb - 0.0000933332591 * T_amb**2   #"Effect of temperature on the capacity of the battery bank from IEEE 485 Table 1"
    high_trip = 0.95 * BattkWh * SOC_frac
 
    #"ENERGY BALANCE ON BATT BANK"
    #"Possible charge/discharge mechanisms accounted for"
    Charge = PV_Batt_Charge_Power + Inv_Batt_dis_P + Gen_Batt_Charge_Power + ORC	
    space_avail = BattkWh*SOC_frac - Batt_SOC_in 	#"kWh"     "remaining capacity to fill"
    
    #" if there are no batteries, all energy gets dumped "
    if BattkWh == 0:
        Charge = 0
        dumpload = PV_Batt_Charge_Power + Gen_Batt_Charge_Power + ORC
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
def storage_levels(Batt_SOC,low_trip,high_trip,BattkWh,TES_SOC,lowT_trip,highT_trip,loadLeft):
 
    #" Battery SOC cases "
    if BattkWh == 0:
        battcase = 1 #"if no batteries, always responds as Low Batt"
    else:
        if Batt_SOC > high_trip:
            battcase = 4
        else:
            if Batt_SOC < low_trip:
                battcase = 1
            else:
                if (Batt_SOC-low_trip) > loadLeft:
                    battcase = 3
                else:
                    battcase = 2
    #This chain of if statements can be cleaned up
 
	#" TES SOC cases "
    if TES_SOC > highT_trip:
        TEScase = 3
    else:
        if TES_SOC < lowT_trip:
            TEScase = 1
        else:
            TEScase = 2
 
    return battcase,TEScase
##=============================================================================================================================

## GetState FUNCTION =====================================================================================================
def getstate(PV_Pow,loadkW,TEScase,battcase,ORC_kW,CSP_A,BattkWh, peakload):

    #" if ORC_kW = 0                             "
    #"  can't be in start state 3, 6, 7, or 8    "
    #" if PV_kW = 0       	     	      	      "
    #" can't be in start_state 2, 5, 7, or 8     "
    #"check if PV power is greater than or less than load demand"							
    
    diff = PV_Pow - loadkW    #"difference between PV output and load"

    #"initializations"
    PV_on = 0
    ORC_on = 0
    Genset_on = 0
    ORCkW_set = np.copy(ORC_kW)
 
    #"PV OUTPUT ON/OFF"
    #" recall low-light cutoff and PV_kW=0 checks in Solar Ambient"

    if PV_Pow > 0:   
        PV_on = 1
    else:
        PV_on = 0
 
    #" THE PV CASES "
    if PV_on == 1: 
        if battcase == 1 and diff < 0:
            Genset_on = 1
            if CSP_A == 0 and ORC_kW > 0:
                #" ORC bottomoing on genset, no TES in system "
                ORC_on = 1
                ORCkW_set = 0.14 * peakload
            if TEScase == 3 and ORC_kW > 0 and BattkWh == 0:
                ORC_on = 1
        else:
            Genset_on = 0
            #" could omit this case if vars initialized to zero above"

    #" THE NO PV CASES "
    if PV_on == 0: 
        if battcase == 4 or battcase == 3:
            #" enough energy in batteries to supply load "
            #" (implements the SMART charging) "
            ORC_on = 0
            Genset_on = 0	     
            #" could omit this case if vars initialized to zero above"
        else:
            #" battery is low OR too low to supply load left"
            if ORC_kW == 0 or (CSP_A > 0 and TEScase == 1):
                #" there is no ORC, or the TES is too low to run the ORC "
                Genset_on = 1
                ORC_on = 0
            else:
                if TEScase== 1 or BattkWh == 0:
                    #" note: catches only the CSP_A=0 cases "
                    #" ORC bottomoing on genset, no TES in system "
                    Genset_on = 1
                    ORC_on = 1
                    ORCkW_set=0.14*peakload
                else:
                    #" adequate charge in TES"
                    if battcase == 1:
                        #" battery is too low to power load from ORC "
                        Genset_on = 1
                        ORC_on = 0
                    else:
                        Genset_on = 0
                        ORC_on = 1
 
    #" Catch cases where TES is over-full but batteries are not "
    #" NOTE: never have TES w/out ORC "
    if TEScase == 3 and battcase < 4 and BattkWh > 0:
        #" charging batteries instead of de-focusing "
        ORC_on = 1
 
    #" set state based on PV_on, CSP_on, Genset_on "
    if PV_on == 1:
        if ORC_on == 1:
            if Genset_on == 1:
                current_state = 7
            else: #"Genset_on=0"
                current_state = 8
        else: #"ORC_on=0"
            if Genset_on == 1:
                current_state = 5
            else: #"Genset_on=0"
                current_state = 2
    else: #"PV_on=0"
        if ORC_on == 1:
            if Genset_on == 1:
                current_state = 6
            else: #"Genset_on=0"
                current_state = 3
        else: #"ORC_on=0"
            if Genset_on == 1:
                current_state = 4
            else: #"Genset_on=0"
                current_state = 1
 
    return current_state, ORCkW_set
##=============================================================================================================================    


## Tes_Energy_balance FUNCTION =====================================================================================================
def tes_energy_balance(highT_trip, TES_SOC_in,TESkWh,CSP_Avail_Pow,timestep,T_amb,Genset_on,Genset_peak):
 
    #"TES is modeled as sensible heat storage in a pebble bed with HTF flowing through it.  The calculations defining the losses as a function of storage temperature, ambient temperature and capacity of the tank were performed in a separate EES file 'Pebble_Storage.EES' and the results of a parametric study were processed into a correlation with Eureqa.  Some important assumptions made are as follows:  Tank temperature varies from 150 to 180C, glycol is the HTF, and the tank is a cylinder with a fixed height and varying diameter.  For the purposes of uGrid TES State of Charge is zero at 150C and full at 180C.  Extrapolation to higher or lower temperatures may be possible but using the losses correlation out of range may produce erroneous results - i.e. it is physically possible to run TES SOC down below zero (for e.g. heating applications) but the losses should be checked carefully and if they produce unrealistic results the original EES file should be referenced to increase the data range for the correlation"
    #"ENERGY BALANCE ON TES TANK"
    #"TES State of charge is previous plus heat addition minus heat losses"

    if TESkWh > 0:
        #" If we have TES "
        #" calculate addition of energy from Genset WHR "
        if Genset_on == 1:
            #"Thermal potential for WHR from the genset exhaust derived from LP GENSET.EES model created for the Shell project, fit using Eureqa R2=0.997"
            Genset_Pow = 0.932071139000018 + 0.00521289941525483*T_amb - 0.0801630128318715*TES_SOC_in/TESkWh - 3.87675754014623e-5*T_amb**2
        else:
            Genset_Pow = 0
 
        #"TES cannot be charged beyond full capacity, collectors are defocused if TES is full (above high cutoff) "
        if TES_SOC_in >= highT_trip:
            CSP_Pow = 0		#" thermal addition to TES "
            Genset_Pow = 0	#" thermal addition to TES "
        else:
            CSP_Pow = np.copy(CSP_Avail_Pow)
 
		#" Total thermal energy additions to TES "
        ChargeTES = CSP_Pow + Genset_Pow
 
        #"TES losses as a function of T_amb and TES SOC (as a proxy for average Tank T) extracted from a dataset exercising the Schuman model (10-node) for a TES of up to 40 MWh using Eureqa"
        TES_loss = 0.000831064446466728*TESkWh + 3.49875188913509*np.arccosh(2.0551369082496 + 0.00148277899794591*TESkWh) - 4.37187301750623 - 7.37880204435673e-6*TESkWh*T_amb - 2.14136361650572e-9*TESkWh^2
        TES_SOC_out = TES_SOC_in+(ChargeTES-TES_loss)*(timestep)   
        TES_frac = TES_SOC_out/TESkWh		#" needed to do ORC efficiency calculations "
    else:
        ChargeTES = 0
        TES_SOC_out = 0
        TES_loss = 0
        TES_frac = 0
        CSP_Pow = 0
 
    return CSP_Pow,ChargeTES,TES_SOC_out,TES_loss, TES_frac
##============================================================================================================================= 


## Get_ORC_info FUNCTION =====================================================================================================
def get_orc_info(T_amb,TES_frac, peakload):
    #" get efficiency characteristics of ORC "
    #" derived from Matthias' thesis variable pinch analysis of ORC - fit using Eureqa "

    Eta_ORC = 0.185897678822732 + 8.7877282742953e-6*T_amb**2 - 0.000297432381152885*TES_frac - 0.000877839693063512*T_amb - 5.07070707070454e-7*T_amb**3 
    ORC_Pow_ratio_TES = 1.1582430818531 + 5.23760049473785e-5*T_amb**2 - 0.00192858794863515*TES_frac - 0.00543790249433032*T_amb - 3.11717171717069e-6*T_amb**3 

    #"Thermal potential for WHR from the genset exhaust derived from LP GENSET.EES model created for the Shell project, fit using Eureqa R2=0.997" 
    ORC_Pow_noTES = peakload * (0.146775411598832 + 2.66715750826574e-7*T_amb**3 - 2.25242051687131e-5*T_amb**2)
 
    return ORC_Pow_ratio_TES, ORC_Pow_noTES, Eta_ORC
##=============================================================================================================================     


## setVars FUNCTION =====================================================================================================
def setvars(highT_trip, current_state,I_b,lowlightcutoff,TES_SOC_in,TESkWh,PV_Avail_Pow,CSP_Avail_Pow,timestep,T_amb,Genset_peak,loadkW,BattkWh,Limit,ORCkW):
    #" Set system variable values based on current state number "
 
    #" initialize "
    Eta_ORC = -1
 
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
        Genset_on = 0
   
        #"ORC + thermal energy usage"
        CSP_Pow,ChargeTES,TES_SOC_out,TES_loss,TES_frac = tes_energy_balance(highT_trip, TES_SOC_in,TESkWh,CSP_Avail_Pow,timestep,T_amb,Genset_on,Genset_peak)
        ORC=0
        ORC_Batt_Charge_Power = 0
        nextTES_SOC = np.copy(TES_SOC_out)

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
        Genset_on = 0
        CSP_Pow,ChargeTES,TES_SOC_out,TES_loss,TES_frac = tes_energy_balance(highT_trip, TES_SOC_in,TESkWh,CSP_Avail_Pow,timestep,T_amb,Genset_on,Genset_peak)
 
        #"ORC + thermal energy usage"
        ORC=0
        ORC_Batt_Charge_Power = 0
        nextTES_SOC = np.copy(TES_SOC_out)

 
    if current_state == 3:
        #" PV, ORC, Genset = 0 1 0 "
        #" ALWAYS have batteries in this state !!!  ORC cannot run alone w/out batteries "
        PV_Pow = 0
        PV_Batt_Charge_Power=0  #"there is no power from the PV"
 
        genload = 0 #" genset is not on "
        Gen_Batt_Charge_Power = 0
        Genset_on = 0
 
        #"ORC + thermal energy usage"
        CSP_Pow,ChargeTES,TES_SOC_out,TES_loss,TES_frac = tes_energy_balance(highT_trip, TES_SOC_in,TESkWh,CSP_Avail_Pow,timestep,T_amb,Genset_on,Genset_peak)
        ORC_Pow_ratio_TES, ORC_Pow_noTES, Eta_ORC = get_orc_info(T_amb, TES_frac, Genset_peak)
 
        if TESkWh > 0:
            ORC=ORCkW*ORC_Pow_ratio_TES
            nextTES_SOC = TES_SOC_out-ORC*(timestep)/Eta_ORC
        else:
            ORC= np.copy(ORC_Pow_noTES)
            nextTES_SOC = np.copy(TES_SOC_out)
            #" load is powered from the ORC/inverter - excess goes to battery charging "
            if ORC - loadkW > 0: #"if the difference is positive there is power available"
                ORC_Batt_Charge_Power = ORC - loadkW   #"ORC charging batteries"
                Inv_Batt_Dis_P = 0	    #"ORC is powering entire load"
            else: 
                ORC_Batt_Charge_Power=0  #"there is no power surplus from the ORC"
                #"load is powered from the inverter -- unless batteries are low (condition checked when state is set)"
                Inv_Batt_Dis_P = ORC - loadkW

 
 
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

        #"ORC + thermal energy usage"
        Genset_on = 1
        CSP_Pow,ChargeTES,TES_SOC_out,TES_loss,TES_frac = tes_energy_balance(highT_trip, TES_SOC_in,TESkWh,CSP_Avail_Pow,timestep,T_amb,Genset_on,Genset_peak)
        ORC=0
        ORC_Batt_Charge_Power = 0
        nextTES_SOC = np.copy(TES_SOC_out)

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
 
        #"ORC + thermal energy usage"
        Genset_on = 1
        CSP_Pow,ChargeTES,TES_SOC_out,TES_loss,TES_frac = tes_energy_balance(highT_trip, TES_SOC_in,TESkWh,CSP_Avail_Pow,timestep,T_amb,Genset_on,Genset_peak)
        ORC=0
        ORC_Batt_Charge_Power = 0
        nextTES_SOC = np.copy(TES_SOC_out)
 
 
    if current_state == 6:
        #" PV, ORC, Genset = 0 1 1 "
        PV_Pow = 0
        PV_Batt_Charge_Power=0  #"there is no power from the PV"
        Inv_Batt_Dis_P = 0  #"the inverter is not supplying any loads"
 
        #"ORC + thermal energy usage"
        Genset_on = 1
        CSP_Pow,ChargeTES,TES_SOC_out,TES_loss,TES_frac = tes_energy_balance(highT_trip, TES_SOC_in,TESkWh,CSP_Avail_Pow,timestep,T_amb,Genset_on,Genset_peak)
        ORC_Pow_ratio_TES, ORC_Pow_noTES, Eta_ORC = get_orc_info(T_amb,TES_frac, Genset_peak)
 
        if TESkWh > 0:
            ORC=ORCkW*ORC_Pow_ratio_TES
            nextTES_SOC = TES_SOC_out-ORC*(timestep)/Eta_ORC
        else:
            ORC = np.copy(ORC_Pow_noTES)
            nextTES_SOC = np.copy(TES_SOC_out)
 
        #" load is powered from the ORC - then generator - if excess, goes to battery charging "
        if ORC > loadkW:
            ORC_Batt_Charge_Power = ORC - loadkW
            genload = 0
        else:
            ORC_Batt_Charge_Power = 0
            genload = loadkW - ORC
 
        #"Determine whether and how much the genset is charging the battery bank based on presence of a battery bank, residual genset capacity, and battery charge rate limit"
        if BattkWh > 0:  #"IF there is a battery and"
            if genload < Genset_peak:  
                #"if loads fall below its peak output the generator will charge the battery bank up to the full charge rate"	
                Difference = Genset_peak-genload   #"residual genset capacity"
                if Difference > (Limit - ORC_Batt_Charge_Power):
                    #"if capacity is higher than the charge rate limit, the genset charges at the limit"
                    Gen_Batt_Charge_Power=Limit - ORC_Batt_Charge_Power
                else:
                    #"the genset charges up to its full residual capacity after supplying the load"
                    Gen_Batt_Charge_Power = np.copy(Difference)  
 
                genload = genload + Gen_Batt_Charge_Power
            else:
                #" supplying peak load "
                Gen_Batt_Charge_Power = 0	

        else:
            Gen_Batt_Charge_Power=0

 
 
    if current_state == 7:
        #" PV, ORC, Genset = 1 1 1 "
        #" being in this state means battery is low AND PV inadequate to power load "
 
        PV_Pow = np.copy(PV_Avail_Pow)
        PV_Batt_Charge_Power=0  #"there is no power surplus from the PV"
        Inv_Batt_Dis_P = 0  #"the inverter is not supplying any loads "
        Genset_on = 1
 
        #"ORC + thermal energy usage"
        CSP_Pow,ChargeTES,TES_SOC_out,TES_loss,TES_frac = tes_energy_balance(highT_trip, TES_SOC_in,TESkWh,CSP_Avail_Pow,timestep,T_amb,Genset_on,Genset_peak)
        ORC_Pow_ratio_TES, ORC_Pow_noTES, Eta_ORC = get_orc_info(T_amb, TES_frac, Genset_peak)
 
        if TESkWh > 0:
            ORC=ORCkW*ORC_Pow_ratio_TES
            nextTES_SOC = TES_SOC_out-ORC*(timestep)/Eta_ORC
        else:
            ORC = np.copy(ORC_Pow_noTES)
            nextTES_SOC = np.copy(TES_SOC_out)
 
        #" excess load is powered from the ORC - then excess goes to battery charging "
        #" if PV+ORC is not supplying entire load, then supply rest from genset "
        #" if supplying any from genset, then check for battery & use peak genset output "
 
        #"in this state, diff is ALWAYS negative --> the load is more than the PV can supply"
        if ORC > -diff:
            #"ORC has excess available beyond supplying load"
            ORC_Batt_Charge_Power = ORC + diff
            genload = 0
        else:
            #"PV + ORC not enough to supply load, also need generator"
            ORC_Batt_Charge_Power = 0
            genload = loadkW - ORC - PV_Pow
 
        #"Determine whether and how much the genset is charging the battery bank based on presence of a battery bank, residual genset capacity, and battery charge rate limit"
        if BattkWh > 0:  #"IF there is a battery and"
            if genload < Genset_peak:  
                #"if loads fall below its peak output the generator will charge the battery bank up to the full charge rate"	
                Difference = Genset_peak-genload   #"residual genset capacity"
                if Difference > (Limit - ORC_Batt_Charge_Power):
                    #"if capacity is higher than the charge rate limit, the genset charges at the limit"
                    Gen_Batt_Charge_Power=Limit - ORC_Batt_Charge_Power
                else:
                    #"the genset charges up to its full residual capacity after supplying the load"
                    Gen_Batt_Charge_Power = np.copy(Difference)  
 
                genload = genload + Gen_Batt_Charge_Power
            else:
                #" supplying peak load "
                Gen_Batt_Charge_Power = 0	

        else:
            Gen_Batt_Charge_Power=0

 
 
    if current_state == 8:
        # " PV, ORC, Genset = 1 1 0"
        #" in this state TES is over temperature... but NOT over-battery "
        #" battery exists, and EITHER Pv is inadequate OR battery is low (not both) "
 
        PV_Pow = np.copy(PV_Avail_Pow)
 
        genload = 0
        Gen_Batt_Charge_Power = 0
        Genset_on = 0
 
        #"ORC + thermal energy usage"
        CSP_Pow,ChargeTES,TES_SOC_out,TES_loss,TES_frac = tes_energy_balance(highT_trip, TES_SOC_in,TESkWh,CSP_Avail_Pow,timestep,T_amb,Genset_on,Genset_peak)
        ORC_Pow_ratio_TES, ORC_Pow_noTES, Eta_ORC = get_orc_info(T_amb, TES_frac, Genset_peak)
 
        if TESkWh > 0:
            ORC=ORCkW*ORC_Pow_ratio_TES
            nextTES_SOC = TES_SOC_out-ORC*(timestep)/Eta_ORC
        else:
            ORC = np.copy(ORC_Pow_noTES)
            nextTES_SOC = np.copy(TES_SOC_out)
 
        #" excess load is powered from the ORC - then excess goes to battery charging "
        if diff > 0: #"if the difference is positive there is power available"
            PV_Batt_Charge_Power = np.copy(diff)   #"PV charging batteries"
            Inv_Batt_Dis_P = 0
            ORC_Batt_Charge_Power = np.copy(ORC)
        else: #"the difference is negative, the load is more than the PV can supply"
            PV_Batt_Charge_Power=0  #"there is no power surplus from the PV"
            if ORC > -diff:
                #"ORC has excess available beyond supplying load"
                ORC_Batt_Charge_Power = ORC + diff
                Inv_Batt_Dis_P = 0
            else:
                #"PV + ORC not enough to supply load, also need inverter"
                ORC_Batt_Charge_Power = 0
                Inv_Batt_Dis_P = diff+ORC
 
 
    return PV_Pow, CSP_Pow,ChargeTES,TES_SOC_out,TES_loss,PV_Batt_Charge_Power,Inv_Batt_Dis_P,genload,Gen_Batt_Charge_Power,ORC,nextTES_SOC,Eta_ORC,ORC_Batt_Charge_Power

##============================================================================================================================= 


## fuel_calcs FUNCTION =====================================================================================================
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
def solar_ambient(th_Hour,Month,NOCT,lat_factor,Pmax_Tco,PVkW,TES_SOC,TESkWh,CSP_A,lowlightcutoff,ORCkW,MSU_TMY):
#month is not used
    #"SOLAR - AMBIENT"
    #"determine ambient conditions and perform the calculations related to irradiance and temperature"
    #"Set beam radiation equal to the irradiance distribution calculated in the solar module"

    #Hour = MSU_TMY.at[:,'Hour'] #for single value
    Hour = MSU_TMY.loc[:,'Hour']  #collects entire column
    Tamb = MSU_TMY.loc[:,'Tamb'] 
    Itrack = MSU_TMY.loc[:,'Itrack']

    #"derive T_amb from TMY dataset"
    T_amb = np.interp(th_Hour, Hour, Tamb) #('MSU_TMY','Tamb','Hour',Hour=th_Hour)	 
	
    if CSP_A > 0 or PVkW > 0:
        #"derive I_b from DNI dataset or modified dataset for tracker"
        #I_b=interpolate('MSU_TMY','Itrack','Hour',Hour=th_Hour)
        I_b = np.interp(th_Hour, Hour, Itrack)  
    else:
        I_b=-999
 
    if I_b > lowlightcutoff: 
        #"IF CSP in equipment"
        if CSP_A > 0:			 
            #"CSP Calculations"
            T_HTF_col_su=150+30*(TES_SOC/TESkWh)  #"Temperature of the collector inlet is assumed proportional to the state of charge of the sensible heat storage ranging up to 200C"
            #"CSP Efficiency based on SORCE2.1_collector_eckerd.EES (Table 3) exercised across a range of T_amb, T_HTF_su and I_b parameters (constant vwind), functionalized in Eureqa R2=0.995"
            PTC_ETA = 0.734327868107202 + (6.93950871301196 + 0.252496799296039*T_amb - 0.320896097990017*T_HTF_Col_su)/I_b
 
            CSP_Avail_Pow=CSP_A*I_b*PTC_ETA/1000   #"CSP kW output, simplified CSP efficiency based on TES_SOC assuming range 150-200C"
		
        else:
            CSP_Avail_Pow = 0
 
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
        CSP_Avail_Pow = 0
        PV_Avail_Pow = 0


    return I_b,T_amb,PV_Avail_Pow,CSP_Avail_Pow
##=============================================================================================================================  


## operation FUNCTION =====================================================================================================
def operation(Month, Pmax_Tco, NOCT, smart, CSP_A, ORCkW, PVkW, TESkWh, BattkWh, peakload, LoadKW_MAK, FullYearEnergy, final, MSU_TMY):
# Month$ was replaced with Month_
    if CSP_A > 0 and ORCkW == 0: 
        Propane=-9999
        print "Warning: ORCKW CANNOT BE ZERO WHEN CSP_A>0 - An ORC is required when CSP is in the loop --> ERROR!!!"

    #not needed
    #hmax=10 #"arbitrary initialization, needed for EES to compile" 

    timestep=1 #"hours - initialize timestep to 10 minutes"
 
    #"Parameterizations by month"
 
    if Month == 1:
        Month_ ='JAN'
        hmax = int(round(31*24/timestep))
        Hour = np.zeros(hmax)
        Hour[0]=0  #"initialize time to the first hour of the month"
 
    if Month == 2:
        Month_='FEB'
        hmax=int(round(28*24/timestep))
        Hour = np.zeros(hmax)
        Hour[0]=744  #"initialize time to the first hour of the month"
 
    if Month == 3:
        Month_='MAR'
        hmax=int(round(31*24/timestep))
        Hour = np.zeros(hmax)
        Hour[0]=1416  #"initialize time to the first hour of the month"
 
    if Month == 4:
        Month_='APR'
        hmax=int(round(30*24/timestep))
        Hour = np.zeros(hmax)
        Hour[0]=2160  #"initialize time to the first hour of the month"
 
    if Month == 5:
        Month_='MAY'
        hmax=int(round(31*24/timestep))
        Hour = np.zeros(hmax)
        Hour[0]=2880  #"initialize time to the first hour of the month"
 
    if Month == 6:
        Month_='JUN'
        hmax=int(round(30*24/timestep))
        Hour = np.zeros(hmax)
        Hour[0]=3624  #"initialize time to the first hour of the month"
 
    if Month == 7:
        Month_='JUL'
        hmax=int(round(31*24/timestep))
        Hour = np.zeros(hmax)
        Hour[0]=4344  #"initialize time to the first hour of the month"
 
    if Month == 8:
        Month_='AUG'
        hmax=int(round(31*24/timestep))
        Hour = np.zeros(hmax)
        Hour[0]=5088  #"initialize time to the first hour of the month"
 
    if Month == 9:
        Month_='SEP'
        hmax=int(round(30*24/timestep)) 
        Hour = np.zeros(hmax)
        Hour[0]=5832  #"initialize time to the first hour of the month"
 
    if Month == 10:
        Month_='OCT'
        hmax=int(round(31*24/timestep)) 
        Hour = np.zeros(hmax)
        Hour[0]=6552  #"initialize time to the first hour of the month"
 
    if Month == 11:
        Month_='NOV'
        hmax=int(round(30*24/timestep))
        Hour = np.zeros(hmax)
        Hour[0]=7296  #"initialize time to the first hour of the month"
 
    if Month == 12:
        Month_ ='DEC'
        hmax=int(round(31*24/timestep))
        Hour = np.zeros(hmax)
        Hour[0]=8016  #"initialize time to the first hour of the month"
 
    #initialize arrays
    State = np.zeros(hmax)
    beam = np.zeros(hmax)  #"to track irradiance"
    T_ambient = np.zeros(hmax)  #"to track ambient temperature"
    PV_Power = np.zeros(hmax)  #"kW electrical"
    CSP_Power = np.zeros(hmax)  #"kW thermal"
    PV_Batt_Charge_Power = np.zeros(hmax)
    ORC_Batt_Charge_Power = np.zeros(hmax)
    ORC_Eta = np.zeros(hmax)
    Inv_Batt_Dis_P = np.zeros(hmax)
    ORC = np.zeros(hmax)
    ChargeTES = np.zeros(hmax)
    TES_SOC = np.zeros(hmax)
    TES_loss = np.zeros(hmax)
    TES_SOC = np.zeros(hmax)
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
    partload=1
    Propane=0
    fuel_kg=0
    Limit = BattkWh/5	#"kW"  "Battery can only be charged at rate up to arbitrary 1/5 of its full capacity rating"
    lowlightcutoff = 100	  #"DNI level under which solar technologies not responsive"
 
    #"battery"
    high_trip = 0.95*BattkWh	  #" kWh "
    low_trip = 0.05*BattkWh	 # " kWh "
    I_b=0
    WHR=0
    genload[0] = 0
    Gen_Batt_Charge_Power[0] = 0
    PV_Batt_Charge_Power[0] = 0
    Inv_Batt_Dis_P[0] =0
 
    #"TES"
    lowT_trip = ORCkW*timestep/0.1
    #" approx amount of energy withdrawn in one timestep if the ORC is ON "
    #" set for minimum possible ORC efficiency of 10% "
    highT_trip = TESkWh*0.95	#"kWh"
 
    #" Initialization of this var is actually no longer necessary "
    current_state = 4 #" initialization state ensures power delivery no matter the available equipment"
 
    #" If battery bank is included in generation equipment "
    if BattkWh > 0: 
        if Month == 1:
            Batt_soc_f=0
            TES_soc_f=0
            Batt_SOC[0]=BattkWh/2 #" Initialize the Batt Bank to half full"
     	else: 
             Batt_final = np.copy(final[0]) #changed from lookup to imported matrix, need to fix column heading
             Batt_SOC[0] = np.copy(Batt_final)
    else:
    	Batt_SOC[0]=0 #"There is no Batt Bank"
 
    #" IF thermal storage is included in the generation equipment "
    if TESkWh > 0:
        if Month == 1: 
            TES_SOC[0]=TESkWh/2  #"Initialize the TES Tank to cold start"
        else: 
            TES_final = np.copy(final[1])  #changed from lookup to imported matrix, need to fix column heading
            TES_SOC[0] = np.copy(TES_final)
    else:
        TES_SOC[0]=0  #"There is no TES Tank"
 
    #"derive factor for latitude tilted PV panel based on I_tilt to DNI ratio from NASA SSE dataset - used for fixed tilt"
    #CALL I_Tilt(Lat, Long, Month:lat_factor)
    lat_factor=1
 
    #"---------------------------------------------------------------------------------------"
    #"WRAPPER LOOP"
    #"---------------------------------------------------------------------------------------"
    Batt_kWh_tot=0  #"dummy variable for cumulative calc at bottom of loop"
 
 
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
            day=math.ceil(Hour[h]/24)
 
        if CSP_A > 0 or PVkW > 0 or ORCkW > 0:
            #" Assess solar availability "
            I_b,T_amb,PV_Avail_Pow,CSP_Avail_Pow = solar_ambient(Hour[h],Month_,NOCT,lat_factor,Pmax_Tco,PVkW,TES_SOC[h],TESkWh,CSP_A,lowlightcutoff, ORCkW, MSU_TMY)
        else:
            T_amb=-9999
            PV_Avail_Pow=0
            CSP_Avail_Pow=0
            I_b=-9999
	
 
        #"LOAD (kW) for this timestep"
        #"calculate the row value in the Nkautest simulated load dataset (based on Hobhouse NRS) corresponding to the simulation timestep"
        HH=205
        Load_Clinic=0.5*(0.690641672 - 0.11362214*dayHour[h] - 0.0000869340922*dayHour[h]**2 + 0.00596194972*dayHour[h]**3 - 0.000750723835*dayHour[h]**4 + 0.0000343476381*dayHour[h]**5 - 5.45160266E-07*dayHour[h]**6)
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
        battcase,TEScase = storage_levels(Batt_SOC[h],low_trip,high_trip,BattkWh,TES_SOC[h],lowT_trip,highT_trip,loadLeft[h])
        
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
        current_state, ORCkW_set = getstate(PV_Avail_Pow,LoadkW[h],TEScase,battcase,ORCkW,CSP_A,BattkWh, peakload)
        ORC_kW=ORCkW_set
 
        #" Set system variables "
        PV_Pow, CSP_Pow, ChTES, TES_SOC_out,TESls,PV_BCP,Inv_BDP,genL,Gen_BCP,ORC_out,nextTES_SOC,eta,ORC_BCP = setvars(highT_trip, current_state,I_b,lowlightcutoff,TES_SOC[h],TESkWh,PV_Avail_Pow,CSP_Avail_Pow,timestep,T_amb,peakload,LoadkW[h],BattkWh,Limit,ORCkW)
 
        #" Balance energy storage in batteries "
        Positive_Charge,Charge_,Batt_SOC_out,dump,high_trip = batt_bal(T_amb, Limit, PV_BCP,Inv_BDP,Gen_BCP,ORC_BCP,Batt_SOC[h],BattkWh,timestep) 
 
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
        CSP_Power[h] = np.copy(CSP_Pow)  #"kW thermal"
        #CSP_Available[h]=CSP_Avail_Pow
        PV_Batt_Charge_Power[h] = np.copy(PV_BCP)
        ORC_Batt_Charge_Power[h] = np.copy(ORC_BCP)
        ORC_Eta[h] = np.copy(eta)
        Inv_Batt_Dis_P[h] = np.copy(Inv_BDP)
        ORC[h] = np.copy(ORC_out)
 
        #" TES "
        ChargeTES[h] = np.copy(ChTES)
        TES_SOC[h] = np.copy(TES_SOC_out)
        TES_loss[h] = np.copy(TESls)
        if h < hmax-1:
            TES_SOC[h+1] = np.copy(nextTES_SOC)
 
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
 
 
    #"Create strings for use as headers in the exported simulation data file"
    #will save data to excel in different way
 
    #not currently needed
    #Hour_ ='Hour'
    #Batt_SOC_ ='Batt_SOC'
    #TES_SOC_ ='TES_SOC'
    #genload_ ='genload'
    #Gen_Batt_Charge_Power_ ='Gen_Batt_Charge_Power'
    #PV_Batt_Charge_Power_ ='PV_Batt_Charge_Power'
    #ORC_Batt_Charge_Power_ ='ORC_Batt_Charge_Power'
    #Inv_Batt_Dis_P_ ='Inv_Batt_Dis_P'
    #PV_Power_ ='PV_Power'
    #CSP_Power_ ='CSP_Power'
    #loadkW_ =' loadkW'
    #beam_ ='beam'
    #CSP_A_ ='CSP_A'
    #ORCkW_ ='ORCkW'
    #PVkW_ ='PVkW'
    #TESkWh_ ='TESkWh'
    #BattkWh_ ='BattkWh'
    #TES_loss_ ='TES_loss'
    #Propane_ ='Propane'
    #State_ ='State'
 
    #this line is to send data to excel, can do something different
    #$Export  /A  /H /N  /Q   'C:\EES_ALL\fixme1.txt'  Month$, CSP_A$, CSP_A, ORCkW$, ORCkW, PVkW$, PVkW, TESkWh$, TESkWh, BattkWh$, BattkWh, Propane$, Propane, Hour$, Hour[1..186], Batt_SOC$, Batt_SOC[1..186], TES_SOC$, TES_SOC[1..186], TES_loss$, TES_loss[1..186], genload$, genload[1..186], Gen_Batt_Charge_Power$, Gen_Batt_Charge_Power[1..186], PV_Batt_Charge_Power$, PV_Batt_Charge_Power[1..186], ORC_Batt_Charge_Power$, ORC_Batt_Charge_Power[1..186], Inv_Batt_Dis_P$, Inv_Batt_Dis_P[1..186], PV_Power$, PV_Power[1..186], CSP_Power$, CSP_Power[1..186], loadkW$, loadkW[1..186], beam$, beam[1..186], state$, State[1..186]}
 
    if Month == 1 or Month == 3 or Month == 5  or Month == 7 or Month == 8 or Month == 10 or Month == 12:
        Batt_SOC_f = np.copy(Batt_SOC[744-1])
        TES_SOC_f = np.copy(TES_SOC[744-1])
 
    if Month == 2: 
        Batt_SOC_f = np.copy(Batt_SOC[672-1])
        TES_SOC_f = np.copy(TES_SOC[672-1])
 
    if Month == 4 or Month == 6 or Month == 9 or Month == 11:
        Batt_SOC_f = np.copy(Batt_SOC[720-1])
        TES_SOC_f = np.copy(TES_SOC[720-1])
 
    final[0] = np.copy(Batt_SOC_f) #changed from copytolookup to inputting to matrix
    final[1] = np.copy(TES_SOC_f) #changed from copytolookup to inputting to matrix
 
    return Propane, DNI, Batt_kWh_tot, final, Batt_SOC, Charge, State, LoadkW

##============================================================================================================================= 

## Tech_total function =========================================================================================================
def Tech_total(BattkWh_Parametric,PVkW_Parametric, CSP_A_Parametric, ORCkW_Parametric, TES_ratio_Parametric,Month_Parametric):

    #"User defined parameters" 
    NOCT=45 #"[C]"  "Nominal Open Circuit Temperature of the PV panel"
    Pmax_Tco=-0.4  #"[%/C]"
    smart=1  #"use a charging strategy that attempts to minimize the usage of the genset"
    peakload=40 #"maximum power output of the load curve [kW]"

    #"variables scaled to the load peak - should be automatically updated with peakload"
    #used with parametric tables
    Month = np.copy(Month_Parametric) #"THIS LINE IS CHANGED WHEN CODE IS BROKEN UP TO RUN ON MULTIPLE PROCESSORS"
    BattkWh=BattkWh_Parametric*peakload  #"[kWh]"
    PVkW=PVkW_Parametric*peakload  #"[kW]"
    CSP_A=CSP_A_Parametric*peakload
    ORCkW=ORCkW_Parametric*peakload
    TES_ratio = np.copy(TES_ratio_Parametric)
    TESkWh=CSP_A*5*TES_ratio #"TES needs to be about 5 times the square meterage of CSP to avoid defocusing, but this can be adjusted via TES_ratio"

    #Load excel files containing LoadKW_MAK, FullYearEnergy, final
    LoadKW_MAK = pd.read_excel('LoadKW_MAK.xlsx',index_col=None, header=None)
    FullYearEnergy = pd.read_excel('FullYearEnergy.xlsx',index_col=None, header=None)
    final = np.array([0,0])
    MSU_TMY = pd.read_excel('MSU_TMY.xlsx')

    Propane, DNI, Batt_kWh_tot, final, Batt_SOC, Charge, State,LoadkW = operation(Month, Pmax_Tco, NOCT, smart, CSP_A, ORCkW, PVkW, TESkWh, BattkWh, peakload, LoadKW_MAK, FullYearEnergy, final, MSU_TMY) 
    
    #These are used for debugging/validating
    #print "CSP_A is " +str(CSP_A)
    #print "DNI is " +str(DNI)

    return Propane, DNI, Batt_kWh_tot, final, Batt_SOC, Charge, State, LoadkW
##=============================================================================================================



#" -----------------------------------------------------------------------------------------------------------"
#"SIMULTANEOUSLY SOLVED EQUATIONS" as a standalone program
#" -----------------------------------------------------------------------------------------------------------"
if __name__ == "__main__":
    #$IFNOT  ParametricTable
    BattkWh_Parametric=7
    PVkW_Parametric=2.36
    CSP_A_Parametric=0
    ORCkW_Parametric=0
    TES_ratio_Parametric=1
    Month_Parametric=6
    #$endif}{$ST$OFF}{$PX$96}
    
    Propane, DNI, Batt_kWh_tot, final, Batt_SOC, Charge, State, LoadkW = Tech_total(BattkWh_Parametric,PVkW_Parametric, CSP_A_Parametric, ORCkW_Parametric, TES_ratio_Parametric,Month_Parametric)