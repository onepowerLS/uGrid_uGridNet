"""
This file contains all the required functions to run economic model standalone, to compare to the EES economic file. 

Created on Tue Jun 21 10:13:16 2016

@author: phylicia Cicilio
"""

from __future__ import division
import numpy as np
import math


## FindTariffLoop FUNCTION =============================================================================================
def FindTariffLoop(lifetime,Revenue,Balance,CashonHand,Cost,loadkWh,tariff):
    for j in range(1,lifetime):
        #"Revenue is a function of the energy supplied to off takers at the tariff rate"
        Revenue[j]= loadkWh * tariff   
        Balance[j] = Revenue[j] - Cost[j]
        CashonHand[j] = CashonHand[j-1] + Revenue[j] - Cost[j]
    return Revenue, Balance, CashonHand
##=====================================================================================================================


## MCASHFLOW FUNCTION =====================================================================================================
def mcashflow (Batt_life_yrs, equity_debt_ratio, term, loadkWh, interest_rate, loanfactor, PVkW, BattKWh, LEC, C1_pv, C1_LPG, Cost_bank, Cost_Propane_yr):
#Removed all thermal system variables and calculations 
    
    lifetime = 15 
    
    #Initialize output variables
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
	
    #"factors for distributing maintenance costs in time as a function of capex see Orosz IMechE"
    f_pv=0.25
    a_pv=0.25
    f=1.25
    a=0.25
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
        tariff = tariff*1.05 #" Increase the tariff until the cash flows are positive "
        for j in range(1,lifetime):
            #"Revenue is a function of the energy supplied to off takers at the tariff rate"
            Revenue[j]= loadkWh * tariff   
            Balance[j] = Revenue[j] - Cost[j]
            CashonHand[j] = CashonHand[j-1] + Revenue[j] - Cost[j]
           

    return LoanPrincipal, year, Cost, Revenue, CashonHand, Balance, M, O, tariff
###================================================================================================		
	
 
## Econ_total function ===========================================================================
def Econ_total(propane, PVkW,BattKWh,Batt_kWh_tot,loanfactor,equity_debt_ratio,LEC):    

    #"Set the financial return and period"
    interest_rate=0.03
    term=12
  
    #"Convert battery throughput into lifetime"
    Batt_lifecycle = 1750 #"The capacity of the battery is approximately 1750 times its C10 rating OPG2-2000.pdf"
    Batt_life_yrs = math.floor((BattKWh*Batt_lifecycle)/(Batt_kWh_tot+0.01))  #"Years of battery life before replacement is necessary, rounded down to an integer"
  
    HH=205 #"number of households in community of interest"
    loadkWh=121217  #"need to update this as integral of load curve"
    peakload=40  #"need to update this from load curve max power"
    node_num = 213  #"from site survey"
    Dist_km = 8.5  #"from ViPor"
    Step_up_Trans_num = 1 #"from ViPor"
    Pole_Trans_num=5 #"from ViPor"
    Pole_num=Dist_km/0.050   #"1 pole for every 50m distribution wire"

    #"Cost functions"
    Cost_Dist_wire = 0.5/1000   #"0.5USD/m"
    Cost_batt=150 #"[$/kWh]"
    Cost_panels=PVkW*450  #"PV Price via Alibaba 2016"
    Cost_control=5000  #"STG Build or PLC"
    Cost_charge_controllers=150*PVkW
    Cost_Pole=40  #"Transmission Pole Prices 2016 from treatedpoles.co.za"
    Cost_Pole_Trans= 150 #"$" "Alibaba 20kVA single phase 11kV/.22kV"
    Cost_Step_up_Trans=1000 #"$" "Alibaba 63kVA single phase 11kV/.22kV"
    Cost_Smartmeter=65*node_num  #"Iometer"
    Cost_MPesa = 70*peakload  #"Estimate for merchant services with vodacom"
    Cost_inv=peakload*800 #"[$/kW peak]"
    Cost_EPC_tracker=200*PVkW
    Cost_EPC_LPG_tank=5000
    Cost_EPC_Power_house=2500
    Cost_EPC_Labor_Plant=14200
    Cost_EPC_Labor_Dist=5500
    Cost_Dev_land=2000
    Cost_Dev_EIA=2000
    Cost_Dev_connection=1000
    Cost_Dev_ICT=3250
    Cost_Dev_contingency=10000
    Cost_Dev_overhead=10000
    Cost_taxes=1500
 
    #"Cost aggregators"
    C1_LPG = (-10354.1143  + 6192.606 * math.log(peakload))   #"Propane Genset costs"  "Based on generac lineup"
 
    Cost_bank = BattKWh * Cost_batt   #"[NREL, USAID Tetratech, health mgmt., PIH] "
 
    Cost_Propane_yr = propane*1.3  #"USD/kg"  "RSA prices 2016" 

    Cost_Dist = Cost_Dist_wire * Dist_km + Cost_Step_up_Trans * Step_up_Trans_num + Cost_Pole_Trans * Pole_Trans_num + Cost_Pole * Pole_num 
 
    Cost_BOS = Cost_bank + Cost_inv + Cost_control + Cost_Dist + Cost_Smartmeter + Cost_MPesa + Cost_charge_controllers   #"Balance of System"
  
    Cost_EPC = Cost_EPC_tracker + Cost_EPC_LPG_tank + Cost_EPC_Power_house + Cost_EPC_Labor_Plant + Cost_EPC_Labor_Dist
 
    Cost_Dev = Cost_Dev_land + Cost_Dev_EIA + Cost_Dev_connection + Cost_Dev_ICT + Cost_Dev_contingency + Cost_Dev_overhead + Cost_taxes
 
    C1_pv = Cost_panels + Cost_BOS + Cost_EPC + Cost_Dev

    LoanPrincipal, year, Cost, Revenue, CashonHand, Balance, M, O, tariff = mcashflow(Batt_life_yrs, equity_debt_ratio, term, loadkWh, interest_rate, loanfactor, PVkW, BattKWh, LEC, C1_pv, C1_LPG, Cost_bank, Cost_Propane_yr)

    #print "Tariff is " + str(tariff)

    return LoanPrincipal, year, Cost, Revenue, CashonHand, Balance, M, O, tariff, Batt_life_yrs     
    
##===============================================================================================
    
#Run economic code standalone
if __name__ == "__main__":

    #These are for running as standalone
    propane=10000
    PVkW=100
    BattKWh=200
    Batt_kWh_tot=50000
    loanfactor=1
    equity_debt_ratio=0
    LEC = 0.1

    LoanPrincipal, year, Cost, Revenue, CashonHand, Balance, M, O, tariff, Batt_life_yrs = Econ_total(propane,PVkW,BattKWh,Batt_kWh_tot,loanfactor,equity_debt_ratio,LEC)

 
#print tariff