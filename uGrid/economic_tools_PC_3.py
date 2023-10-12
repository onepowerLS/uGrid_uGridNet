"""
Economic Tools

This contains all the economic functions which are called by the macro file
for the uGrid tool. 

@author: Phylicia Cicilio
"""

from __future__ import division
import numpy as np
import math
import pandas as pd
from constants import SITE_NAME

sitename = SITE_NAME

## MCASHFLOW FUNCTION =====================================================================================================
def mcashflow(tariff_hillclimb_multiplier,lifetime,f_pv,a_pv,f,a,Batt_life_yrs, equity_debt_ratio, term, loadkWh, interest_rate, loanfactor, PVkW, BattKWh, LEC, C1_pv, C1_LPG, Cost_bank, Cost_Propane_yr):
#This function calculates the tariff based on the inputted equipment sizes and 
# minigrid costs and financial parameters.
    
    #Initialize output variables
    lifetime = int(lifetime)
    LoanPrincipal = np.zeros(lifetime)
    year = np.zeros(lifetime)
    Cost = np.zeros(lifetime)
    Revenue = np.zeros(lifetime)
    CashonHand = np.zeros(lifetime)
    Balance = np.zeros(lifetime)
    M = np.zeros(lifetime)
    O = np.zeros(lifetime)
    tariff = np.copy(LEC) # "USD/kWh tariff for electricity"
    Batt_penalty = 0

    LoanPrincipal[0] = ((C1_pv+C1_LPG)*loanfactor)*(1-equity_debt_ratio)   #"The amount of project finance needed to cover capex and initial opex"
    LoanPrincipal[1] = np.copy(LoanPrincipal[0])
    CashonHand[0] = equity_debt_ratio*((C1_pv+C1_LPG)*loanfactor)+ LoanPrincipal[0]-(C1_pv+C1_LPG)
 
    if interest_rate > 0:
        finance = LoanPrincipal[0]/((1 - (1/(1+interest_rate)**term))/interest_rate)  #"Calculates the amount in each year for loan repayment"
        #print "finance is " + str(finance)
    else:
        finance = 0
    
    if Batt_life_yrs == 0:  #"If the battery needs to be replaced more often than once a year, inflict a heavy penalty"
        Batt_life_yrs = 1
        Batt_penalty = 1
    
    for j in range(1,lifetime):
        #"Maintenance cost is a function of the type of equipment"
        M[j] = (f_pv*a_pv*C1_pv/lifetime)+(f_pv*(1-a_pv)*C1_pv/lifetime**2)*(2*j-1)+(f*a*C1_LPG/lifetime)+(f*(1-a)*C1_LPG/lifetime**2)*(2*j-1)
         
        #"Operating cost is backup fuel - some ITC charges could also be added depending on the metering situation"
        O[j] =  np.copy(Cost_Propane_yr)
          
        #"Financial outlays in each year of operation include loan repayment and O&M"
        Cost[j] = finance + O[j] + M[j]
            
        #"cost of battery replacement every n=batt_life_yrs"
        Modulo = j % Batt_life_yrs #find the remainder
        if Modulo == 0:
            Cost[j] = Cost[j] + Cost_bank + 2 * Cost_bank * Batt_penalty	

        #  "add interest and debit the finance charge from the principal"
        if j > 0:
            LoanPrincipal[j] = LoanPrincipal[j-1] * (1 + interest_rate) - finance

        #"if the loan is paid off THEN there no finance charge"
        if LoanPrincipal[j] <=0:
            Cost[j] = Cost[j] + LoanPrincipal[j]
            LoanPrincipal[j] = 0

    while not all(i > 0 for i in CashonHand[1:]): #continue loop until all values in CashonHand[1:] are greater than 0
        tariff = tariff*tariff_hillclimb_multiplier #" Increase the tariff until the cash flows are positive "
        for j in range(1,lifetime):
            #"Revenue is a function of the energy supplied to off takers at the tariff rate"
            Revenue[j]= loadkWh * tariff   
            Balance[j] = Revenue[j] - Cost[j]
            CashonHand[j] = CashonHand[j-1] + Revenue[j] - Cost[j]
           

    return LoanPrincipal, year, Cost, Revenue, CashonHand, Balance, M, O, tariff
###================================================================================================		

 
## Econ_total function ===========================================================================
def Econ_total(propane_kg, PVkW,BattKWh,Batt_kWh_tot,peakload,loadkWh):    
# This function calls in all the necessary economic inputs from the uGrid_Input.xlsx 
# and calls the mcashflow function to calculate the tariff. 
    
    #Load all Econ input
    Econ_Parameters = pd.read_excel(sitename + '_uGrid_Input.xlsx', sheet_name = 'Econ')
    Size_costing_parameters = pd.read_excel(sitename + '_uGrid_Input.xlsx', sheet_name = 'Sizing_Costing', usecols='F')

    #"factors for distributing maintenance costs in time as a function of capex see Orosz IMechE"
    f_pv= Econ_Parameters['f_pv'][0]
    a_pv=Econ_Parameters['a_pv'][0]
    f=Econ_Parameters['f'][0]
    a=Econ_Parameters['a'][0]    
    
    #"Set the financial return and period"
    interest_rate=Econ_Parameters['interest_rate'][0]
    term=Econ_Parameters['term'][0]
    loanfactor=Econ_Parameters['loanfactor'][0]
    equity_debt_ratio=Econ_Parameters['equity_debt_ratio'][0]
    lifetime = Econ_Parameters['lifetime'][0]
    tariff_hillclimb_multiplier = Econ_Parameters['tariff_hillclimb_multiplier'][0]
      
    #"Convert battery throughput into lifetime"
    Batt_life_yrs = np.floor((BattKWh*Econ_Parameters['Batt_lifecycle'][0])/(Batt_kWh_tot+0.01))  #"Years of battery life before replacement is necessary, rounded down to an integer"

    #"Cost functions"  
    #Pole_num=Econ_Parameters['Dist_km'][0] /0.050   #"1 pole for every 50m distribution wire"
    Cost_panels=PVkW*Size_costing_parameters.iloc[0]  #"PV Price via Alibaba 2016"
    #Cost_charge_controllers=Econ_Parameters['Cost_charge_controllers_per_kW'][0]*PVkW
    #Cost_Smartmeter=65*Econ_Parameters['node_num'][0]  #"Iometer"
    Cost_MPesa = Econ_Parameters['Cost_Mpesa_per_kWLoad'][0]*peakload  #"Estimate for merchant services with vodacom"
    Cost_inv=peakload*Size_costing_parameters.iloc[2] #"[$/kW peak]"
    Cost_EPC_tracker=Size_costing_parameters.iloc[3]*PVkW
    Cost_BOS = Size_costing_parameters.iloc[5]*PVkW 
    Cost_reticulation = Size_costing_parameters.iloc[6]
 
    #"Cost aggregators"
    C1_LPG = Size_costing_parameters.iloc[4]*peakload   #"Propane Genset costs"  "Based on generac lineup"
 
    Cost_bank = BattKWh * Size_costing_parameters.iloc[1]   #"[NREL, USAID Tetratech, health mgmt., PIH] "
 
    Cost_Propane_yr = propane_kg*1.3  #"USD/kg"  "RSA prices 2016"

#    Cost_Dist = Econ_Parameters['Cost_Dist_wire'][0] * Econ_Parameters['Dist_km'][0] + Econ_Parameters['Cost_Step_up_Trans'][0] * Econ_Parameters['Step_up_Trans_num'][0] + Econ_Parameters['Cost_Pole_Trans'][0] * Econ_Parameters['Pole_Trans_num'][0] + Econ_Parameters['Cost_Pole'][0] * Pole_num 
    Cost_Dist = Cost_reticulation 
 
    #Cost_BOS = Cost_bank + Cost_inv + Econ_Parameters['Cost_control'][0] + Cost_Dist + Cost_Smartmeter + Cost_MPesa + Cost_charge_controllers   #"Balance of System"
    #Cost_BOS = Econ_Parameters['Cost_control'][0] + Cost_charge_controllers   #"Balance of System"
  
    #Cost_EPC = Cost_EPC_tracker + Econ_Parameters['Cost_EPC_LPG_tank'][0] + Econ_Parameters['Cost_EPC_Power_house'][0] + Econ_Parameters['Cost_EPC_Labor_Plant'][0] + Econ_Parameters['Cost_EPC_Labor_Dist'][0]
    Cost_EPC = 0.1* (Cost_Dist + Cost_BOS + Cost_bank + C1_LPG + Cost_inv + Cost_panels + Cost_EPC_tracker)
    #Cost_Dev = Econ_Parameters['Cost_Dev_land'][0] + Econ_Parameters['Cost_Dev_EIA'][0] + Econ_Parameters['Cost_Dev_connection'][0] + Econ_Parameters['Cost_Dev_ICT'][0] + Econ_Parameters['Cost_Dev_contingency'][0] + Econ_Parameters['Cost_Dev_overhead'][0] + Econ_Parameters['Cost_taxes'][0]
 
    #C1_pv = Cost_panels + Cost_BOS + Cost_EPC + Cost_Dev
    #print('Cost EPc '+str(Cost_EPC),'Cost Distribution'+ str(Cost_Dist),'Cost BOS' +str(Cost_BOS),'Cost bank' +str(Cost_bank),'Cost Genset' + str(C1_LPG),'Cost inverter'+ str(Cost_inv),'Cost panels'+ str(Cost_panels),'Cost tracker'+ str(Cost_EPC_tracker))
    C1_pv = Cost_EPC + Cost_Dist + Cost_BOS + Cost_bank + C1_LPG + Cost_inv + Cost_panels + Cost_EPC_tracker
    #print('C1_pv is :' + str(C1_pv))
    Cost_labour = Econ_Parameters['Cost_EPC_Labor_Dist'][0]
    
    LEC = 0.1 #this is a starting point for LEC. This could potentially be done without a hill-climb and be directly solved

    LoanPrincipal, year, Cost, Revenue, CashonHand, Balance, M, O, tariff = mcashflow(tariff_hillclimb_multiplier,lifetime,f_pv,a_pv,f,a,Batt_life_yrs, equity_debt_ratio, term, loadkWh, interest_rate, loanfactor, PVkW, BattKWh, LEC, C1_pv, C1_LPG, Cost_bank, Cost_Propane_yr)

    #print "Tariff is " + str(tariff)

    return LoanPrincipal, year, Cost, Revenue, CashonHand, Balance, M, O, tariff, Batt_life_yrs, Cost_EPC, Cost_Dist, Cost_BOS, Cost_bank, C1_LPG, Cost_inv, Cost_panels, Cost_EPC_tracker, Cost_labour, C1_pv, PVkW
    
##===============================================================================================
    
#Run economic code standalone
if __name__ == "__main__":
# This python file can be run as standalone. In order to run as a standalone 
# all the inputs for the Econ_Total function need to be specified. 

    #These are for running as standalone
    propane_kg=10000
    PVkW=100
    BattKWh=200
    Batt_kWh_tot=50000
    LoadKW_MAK = pd.read_excel('LoadKW_MAK.xlsx',index_col=None, header=None)
    peakload_buffer = 1.2
    peakload=max(LoadKW_MAK[0])*peakload_buffer
    loadkWh = sum(LoadKW_MAK[0])

    LoanPrincipal, year, Cost, Revenue, CashonHand, Balance, M, O, tariff, Batt_life_yrs = Econ_total(propane_kg,PVkW,BattKWh,Batt_kWh_tot,peakload,loadkWh)

 
