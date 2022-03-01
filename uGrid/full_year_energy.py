"""
Full Year Energy Calculator

This contains functions and algorithms to calculate battery energy demand for each hour of each day of the year when solar PVs are not expected to be generating any electricity (i.e, overning demand). This is commonly known as the full year energy demand. 

@author: Thabo Monoto
"""

import sys
# get_ipython().system('$sys.executable -m pip install numpy pandas matplotlib seaborn requests mat4py openpyxl ')

import os
import pandas as pd
import numpy as np
from itertools import product
import matplotlib as plt
# from IPython.core.debugger import set_trace
import pdb
import itertools as it
import glob
from constants import SITE_NAME

def get_8760(village_name):
    filtered_list = glob.glob(f'{village_name}*8760*.xlsx')
    for f in filtered_list:
        if village_name in f and '8760' in f:
            return f
    return None

def split_vector(vect):
    """ This function splits an 8760 vector into 366 vectors: The first vector has 18 elements, the last vector
        has 6 elements, and the rest have 24 elements. 
        
        Parameters
        ----------
        vect: ndarray
            the vector to be split
        
        Returns
        -------
        vect_out: ndarray 
            output of the splitting process, an array consisting of 366 vectors
    
    """
    
    a, b, c,d = vect[:18], vect[18:8754], [], vect[8754:]
    f = lambda b, n=24: [b[i:i+n] for i in range(0, len(b), n)]
    for i in range(len(f(b))):
        c.append(f(b)[i])
    vect_out = [a] + c + [d]
    
    return vect_out

def full_year_energy_calc(a, b,index):
    """ This function calculates the full year energy (the full year energy 
        shows the decremental, hourly energy requirement when the solar pv is not
        generating (typically at night) ) of an area.
    
        Parameters
        ----------
        a: ndarray
            array of daily energy totals
        b: ndarray
            8760 of the area under consideration
        index: ndarray
            array whose entries function as iteration indices (representing hours in a 'day') 
    
        Returns
        -------
        full_year: ndarray
            full year energy of the area under consideration
    """
    
    #TODO: remove dependency on a template full year enery, require only 8760 to run the algorithm
    full_year = [] 
    b = vect_split(b)
    k = 0
    m = [] #vector of incremental indices
    for i in range(len(a)):
        full_year.append(a[i])
        d = index[i]
        for j in range(d):
            if b[i][j] > 0.0 and j+1 < d and b[i][j+1] !=0.0:
                #pdb.set_trace()
                full_year.append(full_year[k]-b[i][j+1])
                k+=1
            elif b[i][j]<=0.0:
                full_year.append(0.0)
                k+=1
        m.append(k)
        k = m[i]+1
    return pd.DataFrame(full_year)

#convert excel data to a pandas dataframe
#def data_proc
sitename = SITE_NAME
template_full_year_energy = pd.read_excel('full_year_energy.xlsx')
# Load Files
loadfile = get_8760(sitename)
# print(load_file)
_8760 = pd.read_excel(loadfile, sheet_name='8760', usecols ='B')

#convert dataframe to numpy array (results in an array of arrays)
_8760 = _8760.to_numpy()


array = np.delete(template_full_year_energy.to_numpy(),0,1)

# set all nonzero values in template_full_year_energy to 1 and store the results in a new array
array1 =[]
for i in range(len(array)):
    if array[i][0]>0:
        array[i][0] = 1
    else:
        array[i][0] = 0
    np.array(array1.append(array[i][0]))
    
array3 =[]
for i in range(len(_8760)):
    np.array(array3.append(_8760[i][0]))


# create a modified 8760 and populate with the required hourly overnight energy values and zeroes for daytime values  
modified8760 = []

for i in range(len(array1)):
    if array1[i] == 1.0:
        modified8760.append(array3[i])
    else:
        modified8760.append(0.0)

# split the modified 8760 using the vect_split function
split_8760 = split_vector(modified8760)

day_totals =[]
for i in range(len(split_8760)):
    day_totals.append(max(np.cumsum(split8760[i]))

indices = []
for i in range(366):
    indices.append(len(split8760[i]))



