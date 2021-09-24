
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

def vect_split(vect):
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
template_full_year_energy = pd.read_excel('/home/onepower/Downloads/FullYearEnergy.xlsx')
_8760 = pd.read_excel('/home/onepower/Downloads/15062021_1640_SEB_8760_C576.xlsx', sheet_name='8760', usecols= 'B', header=0)
#print(_8760)

#convert dataframe to numpy array (results in an array of arrays)
_8760 = _8760.to_numpy()
#print(_8760)

array = np.delete(template_full_year_energy.to_numpy(),0,1)
#print(array)

# set all nonzero values in template_full_year_energy to 1 and store the results in a new array
array1 =[]
for i in range(len(array)):
    if array[i][0]>0:
        array[i][0] = 1
    else:
        array[i][0] = 0
    np.array(array1.append(array[i][0]))
# 
array3 =[]
for i in range(len(_8760)):
    #array3.append(array2[i][0])
    np.array(array3.append(_8760[i][0]))


# create a modified 8760 and populate with the required hourly overnight energy values and zeroes for daytime values  
modified8760 = []

for i in range(len(array1)):
    if array1[i] == 1.0:
        modified8760.append(array3[i])
    else:
        modified8760.append(0.0)
#print(modified8760)

# split the modified 8760 using the vect_split function
split8760 = vect_split(modified8760)
#print(len(array6[0]))
day_totals =[]
for i in range(len(split8760)):
    day_totals.append(max(np.cumsum(split8760[i])))
#print(day_totals)

indices = []
for i in range(366):
    indices.append(len(split8760[i]))
#print(indices)

#print(full_year_energy_calc(day_totals, modified8760, indices))




