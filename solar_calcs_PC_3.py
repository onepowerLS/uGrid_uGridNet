# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 02:07:35 2018

@author: Phy
"""

from __future__ import division
import numpy as np
import pandas as pd
import math as m

## Solar Calcs from NREL SOLPOS===============================================================
#
def NRELTheta(year,hour,longitude,latitude,timezone):    
# inputs: day number, hour
#/*============================================================================
#*    Local Void function geometry
#*
#*    Does the underlying geometry for a given time and location
#*----------------------------------------------------------------------------*/
#static void geometry ( struct posdata *pdat )
#{
#  float bottom;      /* denominator (bottom) of the fraction */
#  float c2;          /* cosine of d2 */
#  float cd;          /* cosine of the day angle or delination */
#  float d2;          /* pdat->dayang times two */
#  float delta;       /* difference between current year and 1949 */
#  float s2;          /* sine of d2 */
#  float sd;          /* sine of the day angle */
#  float top;         /* numerator (top) of the fraction */
#  int   leap;        /* leap year counter */

# day number (daynum)
    daynum = (hour//24)+1
    
# hour in day
    if hour > 23:
        dayhour = hour - daynum*24 
    else:
        dayhour = hour

#  /* Day angle */
#      /*  Iqbal, M.  1983.  An Introduction to Solar Radiation.
#            Academic Press, NY., page 3 */
#    dayang = 360.0 * ( daynum - 1 ) / 365.0 #radians

#    /* Earth radius vector * solar constant = solar energy */
#        /*  Spencer, J. W.  1971.  Fourier series representation of the
#            position of the sun.  Search 2 (5), page 172 */
#    sd     = m.sin(dayang)
#    cd     = m.cos(dayang)
#    d2     = 2.0 * dayang
#    c2     = m.cos(d2)
#    s2     = m.sin(d2)

#    erv  = (1.000110 + 0.034221 * cd + 0.001280 * sd) + (0.000719 * c2 + 0.000077 * s2)

#    /* Universal Coordinated (Greenwich standard) time */
#        /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
#            approximate solar position (1950-2050).  Solar Energy 40 (3),
#            pp. 227-235. */
    utime = dayhour - timezone

#    /* Julian Day minus 2,400,000 days (to eliminate roundoff errors) */
#        /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
#            approximate solar position (1950-2050).  Solar Energy 40 (3),
#            pp. 227-235. */

#    /* No adjustment for century non-leap years since this function is
#       bounded by 1950 - 2050 */
    delta = year - 1949
    leap = int( delta / 4.0 )
    julday = 32916.5 + delta * 365.0 + leap + daynum + utime / 24.0

#    /* Time used in the calculation of ecliptic coordinates */
#    /* Noon 1 JAN 2000 = 2,400,000 + 51,545 days Julian Date */
#        /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
#            approximate solar position (1950-2050).  Solar Energy 40 (3),
#            pp. 227-235. */
    ectime = julday - 51545.0

#    /* Mean longitude */
#        /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
#            approximate solar position (1950-2050).  Solar Energy 40 (3),
#            pp. 227-235. */
    mnlong  = m.radians(280.460 + 0.9856474 * ectime)

#    /* (dump the multiples of 360, so the answer is between 0 and 360) */
    if mnlong > m.radians(360):
        mnlong -= m.radians(360.0) * int(mnlong / m.radians(360.0) )
    if mnlong < 0 :
        mnlong += m.radians(360.0)

#    /* Mean anomaly */
#        /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
#            approximate solar position (1950-2050).  Solar Energy 40 (3),
#            pp. 227-235. */
    mnanom  = m.radians(357.528 + 0.9856003 * ectime)

#    /* (dump the multiples of 360, so the answer is between 0 and 360) */
    if mnanom > m.radians(360):
        mnanom -= m.radians(360) * int(mnanom / m.radians(360) )
    if mnanom < 0:
        mnanom += m.radians(360)

#    /* Ecliptic longitude */
#        /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
#            approximate solar position (1950-2050).  Solar Energy 40 (3),
#            pp. 227-235. */
    eclong  = mnlong + m.radians(1.915) * m.sin(mnanom) + m.radians(0.020) * m.sin(2.0 * mnanom )

#    /* (dump the multiples of 360, so the answer is between 0 and 360) */
    if eclong > m.radians(360):
        eclong -= m.radians(360) * int(eclong / m.radians(360) )
    if eclong < 0:
        eclong += m.radians(360)

#    /* Obliquity of the ecliptic */
#        /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
#            approximate solar position (1950-2050).  Solar Energy 40 (3),
#            pp. 227-235. */

#   /* 02 Feb 2001 SMW corrected sign in the following line */
#/*  pdat->ecobli = 23.439 + 4.0e-07 * pdat->ectime;     */
    ecobli = m.radians(23.439 - 4.0*10**-7 * ectime)

#    /* Declination */
#        /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
#            approximate solar position (1950-2050).  Solar Energy 40 (3),
#            pp. 227-235. */
    declin = m.asin(m.sin(ecobli) * m.sin(eclong)) #radians

#    /* Right ascension */
#        /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
#            approximate solar position (1950-2050).  Solar Energy 40 (3),
#            pp. 227-235. */
    top = m.cos(ecobli) * m.sin(eclong)
    bottom = m.cos(eclong)

    rascen = m.atan2(top,bottom)

#    /* (make it a positive angle) */
    if rascen < 0:
        rascen += m.radians(360)

#    /* Greenwich mean sidereal time */
#        /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
#            approximate solar position (1950-2050).  Solar Energy 40 (3),
#            pp. 227-235. */
    gmst = 6.697375 + 0.0657098242 * ectime + utime

#    /* (dump the multiples of 24, so the answer is between 0 and 24) */
    if gmst > 23:
        gmst -= 24.0 * int(gmst / 24.0 )
    if gmst < 0:
        gmst += 24.0

#    /* Local mean sidereal time */
#        /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
#            approximate solar position (1950-2050).  Solar Energy 40 (3),
#            pp. 227-235. */
    lmst = gmst * 15.0 + longitude

#    /* (dump the multiples of 360, so the answer is between 0 and 360) */
    if lmst > 360:
        lmst -= 360.0 * int(lmst / 360.0)
    if lmst < 0:
        lmst += 360.0

#    /* Hour angle */
#        /*  Michalsky, J.  1988.  The Astronomical Almanac's algorithm for
#            approximate solar position (1950-2050).  Solar Energy 40 (3),
#            pp. 227-235. */
    hrang = m.degrees(m.radians(lmst) - rascen) #returns in degrees

#    /* (force it between -180 and 180 degrees) */
    if hrang < -180.0:
        hrang += 360.0
    elif hrang > 180.0:
        hrang -= 360.0

    return hrang,declin,daynum


# =============================================================================
# /*============================================================================
# *    Local Void function zen_no_ref
# *
# *    ETR solar zenith angle
# *       Iqbal, M.  1983.  An Introduction to Solar Radiation.
# *            Academic Press, NY., page 15
# *----------------------------------------------------------------------------*/
def SolarZenith(declin,latitude,hrang):
# static void zen_no_ref ( struct posdata *pdat, struct trigdata *tdat )
# {
#   float cz;          /* cosine of the solar zenith angle */
# 
#    /* Earth radius vector * solar constant = solar energy */
#        /*  Spencer, J. W.  1971.  Fourier series representation of the
#            position of the sun.  Search 2 (5), page 172 */
    sd = m.sin(declin)
    cd = m.cos(declin)
#    d2     = 2.0 * dayang
#    c2     = m.cos(d2)
#    s2     = m.sin(d2)
    sl = m.sin(m.radians(latitude))
    cl = m.cos(m.radians(latitude))
    ch = m.cos(m.radians(hrang))
    
    #     localtrig( pdat, tdat );
    cz = sd * sl + cd * cl * ch
# 
#     /* (watch out for the roundoff errors) */
    if abs(cz) > 1.0:
        if cz >= 0.0:
            cz =  1.0
        else:
            cz = -1.0
#     }
# 
    zenetr = m.degrees(m.acos(cz)) 
# 
#     /* (limit the degrees below the horizon to 9 [+90 -> 99]) */
    if zenetr > 99.0:
        zenetr = 99.0;
# 
#    elevetr = 90.0 - pdat->zenetr;
# }
    return zenetr
# =============================================================================


## Theta =====================================================================
def GetTheta(declin,latitude,hrang,slope,azimuth):
    sd = m.sin(declin)
    cd = m.cos(declin)
#    d2     = 2.0 * dayang
#    c2     = m.cos(d2)
#    s2     = m.sin(d2)
    sl = m.sin(m.radians(latitude))
    cl = m.cos(m.radians(latitude))
    ch = m.cos(m.radians(hrang))
    sh = m.sin(m.radians(hrang))
    cb = m.cos(m.radians(slope))
    sb = m.sin(m.radians(slope))
    ca = m.cos(m.radians(azimuth))
    sa = m.sin(m.radians(azimuth))
    
    
    #     localtrig( pdat, tdat );
    ct = sd*sl*cb - sd*cl*sb*ca + cd*cl*cb*ch + cd*sl*sb*ca*ch + cd*sb*sa*sh
    #     /* (watch out for the roundoff errors) */
    if abs(ct) > 1.0:
        if ct >= 0.0:
            ct =  1.0
        else:
            ct = -1.0
#     }
# 
    theta = m.degrees(m.acos(ct)) 
# 
#     /* (limit the degrees below the horizon to 9 [+90 -> 99]) */
    if theta > 99.0:
        theta = 99.0;

    return theta
# =============================================================================
 
## Calc Global Radiation Incident on PV Array ================================
def GetGT(G,slope, pg, theta, zenetr, declin, latitude,daynum):
    if G != 0:
        G = G/1000
        Gsc = 1.367 #the solar constant [kW/m2]
        Gon = Gsc*(1+0.033*m.cos(m.radians(360*daynum/365)))
        Go = Gon*m.cos(m.radians(zenetr))
        kt = G/Go #G is GHI from TMY datasets
        if kt <= 0.22:
            Gd = (1-0.09*kt)*G
        elif kt > 0.22 and kt <= 0.8:
            Gd = (0.9511-0.1604*kt+4.388*kt**2-16.638*kt**3+12.336*kt**4)*G
        elif kt > 0.8:
            Gd = 0.165*G
        Gb = G - Gd   
        Rb = m.cos(m.radians(theta))/m.cos(m.radians(zenetr))
        Ai = Gb/Go
        #Catch roundoff errors, so irradiance stays >=0
        if Gd < 0:
            Gd = 0
        if Gb < 0:
            Gb = 0
        f = m.sqrt(Gb/G)
        Gt = (Gb + Gd*Ai)*Rb + Gd*(1-Ai)*((1+m.cos(m.radians(slope)))/2)*(1+f*(m.sin(m.radians(slope/2))**3))+G*pg*((1-m.cos(m.radians(slope)))/2)
    else:
        Gt = 0
        

    return Gt
##============================================================================

## Get PV Power ===============================================================
def GetPVPower(Ypv,fpv,Gt,alpha_p,Tamb,eff_mpp,f_inv):
    Gt_stc = 1 #incident radiation @ STC [1 kW/m2]
    Gt_noct = 0.8 #solar radiation at which the NOCT is defined [kW/m2]
    Ta_noct = 20 #ambient temperature at which the NOCT is defined [C]
    Tc_stc = 25 #cell temperature under stc [C]
    tau_alpha = 0.9 #suggested value from Duffie and Beckman
    Tc_noct = 25 #nominal operating cell temperature [C]
    #HOMER assumes that the PV array always operates at its maximum power point, as it does when controlled by a maximum power point tracker. That means HOMER assumes the cell efficiency is always equal to the maximum power point efficiency
    #n_mp_stc = Ypv/(Apv*Gt_stc) #would need to calc Apv
    
    #The full calculation of Tc incorporating the calculation for max power point efficiency is not used, as not giving good results. 
    #Tc_top = Tamb + (Tc_noct-Ta_noct)*(Gt/Gt_noct)*(1-(n_mp_stc*(1-alpha_p*Tc_stc))/tau_alpha)
    #Tc_bottom = 1+(Tc_noct-Ta_noct)*(Gt/Gt_noct)*(alpha_p*n_mp_stc/tau_alpha)
    #Tc_full = Tc_top/Tc_bottom

    #Alternate calculation for Tc using assumed version of max power point efficiency
    Tc = Tamb + (Gt/Gt_noct)*(Tc_noct-Ta_noct)*(1-(eff_mpp/tau_alpha))
    Ppv = Ypv*fpv*f_inv*(Gt/Gt_stc)*(1+alpha_p*(Tc-Tc_stc))
    
    if Ppv > Ypv:
        Ppv = np.copy(Ypv)
    elif Ppv < 0:
        Ppv = 0
    
    return Ppv
##=============================================================================
  
## Calculate all solar calcs ==================================================
def SolarTotal(MSU_TMY,year,th_hour,longitude,latitude,timezone,slope,azimuth,pg,Ypv,fpv,alpha_p,eff_mpp,f_inv):

    #Get Values from MSU_TMY
    Hour = MSU_TMY.loc[:,'Hour']  #collects entire column
    Tamb = MSU_TMY.loc[:,'Tamb'] 
    GHI = MSU_TMY.loc[:,'GHI'] #in the TMY spreadsheet DNI, GHI, and itrack are all the same
    G = np.interp(th_hour, Hour, GHI) 
    T_amb = np.interp(th_hour, Hour, Tamb) #('MSU_TMY','Tamb','Hour',Hour=th_Hour)	 

    
    hrang,declin,daynum = NRELTheta(year,th_hour,longitude,latitude,timezone)
    theta = GetTheta(declin,latitude,hrang,slope,azimuth)
    zenetr = SolarZenith(declin,latitude,hrang)
    Gt = GetGT(G,slope, pg, theta, zenetr, declin, latitude,daynum)
    Ppv = GetPVPower(Ypv,fpv,Gt,alpha_p,T_amb,eff_mpp,f_inv)
    
    return m.degrees(hrang),m.degrees(declin),theta,Gt,Ppv,T_amb
##============================================================================= 



if __name__ == "__main__":
    #Location Inputs
    latitude = -33
    longitude = 18
    hour = 10
    slope = 0
    azimuth = 0
    year = 2005
    timezone = 2
    
    #PV inputs/assumptions
    G = 1000 #GHI from TMY datasheets
    pg = 0.20 #ground reflectance, aka albedo [%]. A typical value for grass-covered areas is 20%. Snow-covered areas may have a reflectance as high as 70%. 
    Ypv = 10 #rated capacity of the PV array. power output under stc [kW]
    fpv = 0.90 #PV derating factor [%]
    alpha_p = -0.002 #temperature coefficient of power [%/C]. Average Values: https://www.homerenergy.com/products/pro/docs/3.10/pv_temperature_coefficient_of_power.html
    Tamb = 35
    #n_mp_stc = 0.90 #maximum power point efficienty under stc [%]
    eff_mpp = 0.9
    f_inv = 0.9    
    
    #theta = ThetaCalc(hour,longitude,latitude,slope,azimuth)
    #print "theta is " +str(m.degrees(theta))
    #print "cos of solar zenith is " +str(m.cos(theta))
    hrang,declin,daynum = NRELTheta(year,hour,longitude,latitude,timezone)
    print("hour angle is " +str(hrang))
    print("solar declination is "+str(m.degrees(declin)))
    zenetr = SolarZenith(declin,latitude,hrang)
    print("solar zenith angle is "+str(zenetr))
    theta = GetTheta(declin,latitude,hrang,slope,azimuth)
    print("Theta is "+str(theta))
    Gt = GetGT(G,slope, pg, theta, zenetr, declin, latitude,daynum)
    print("The solar radiation incident on the PC array is "+str(Gt)+" [kW/m2]")
    Ppv = GetPVPower(Ypv,fpv,Gt,alpha_p,Tamb,eff_mpp,f_inv)
    print("The PV Panel Power output is "+str(Ppv)+" [kW]")
    
    
    
    