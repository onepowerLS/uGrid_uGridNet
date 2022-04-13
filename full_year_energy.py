#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
# get_ipython().system('$sys.executable -m pip install numpy pandas matplotlib seaborn requests mat4py openpyxl ')


# In[2]:


import os
import pandas as pd
import numpy as np
from itertools import product
import matplotlib as plt
# from IPython.core.debugger import set_trace
import pdb
import itertools as it


# In[24]:


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
    return full_year


# In[37]:


#convert excel data to a pandas dataframe
temp_fye = pd.read_excel('/home/onepower/Downloads/FullYearEnergy.xlsx')
xxx_8760 = pd.read_excel('/home/onepower/Downloads/02082021_1506_LEB_8760_C077.xlsx', sheet_name='8760', usecols= 'B', header=0)
print(xxx_8760)


# In[38]:


#convert dataframe to numpy array (results in an array of arrays)
xxx_8760 = xxx_8760.to_numpy()
print(xxx_8760)


# In[39]:


array = np.delete(temp_fye.to_numpy(),0,1)
print(array)


# In[40]:


#set all nonzero values in temp_fye to 1 and store the results in a new array
array1 =[]
for i in range(len(array)):
    if array[i][0]>0:
        array[i][0] = 1
    else:
        array[i][0] = 0
    np.array(array1.append(array[i][0]))


# In[41]:


array1


# In[44]:



array3 =[]
for i in range(len(xxx_8760)):
    #array3.append(array2[i][0])
    np.array(array3.append(xxx_8760[i][0]))


# In[45]:


array3
#array4 = np.concatenate(array1,array3)


# In[46]:


#this loop populates array4 with the required hourly (overnight) energy values
array4 = []

for i in range(len(array1)):
    if array1[i] == 1.0:
        array4.append(array3[i])
    else:
        array4.append(0.0)
print(len(array4))


# In[47]:


#array of values to subtract from the daily (overnight) totals
array4


# In[48]:


array5 = []
for i in range(len(array4)):
    if array1 == 0.0:
        array5.append(0.0)
    else:
        array5.append(array4[i])

print(array5)


# In[50]:


array6 = vect_split(array5)
#print(len(array6[0]))
day_totals =[]
for i in range(len(array6)):
    day_totals.append(max(np.cumsum(array6[i])))
print(day_totals)


# In[51]:


indices = []
for i in range(366):
    indices.append(len(array6[i]))
print(indices)


# In[52]:


full_year_energy_calc(day_totals, array4, indices)


# In[23]:


print(len(full_year_energy_calc(array7, array4, array8)))


# In[19]:


documents = [['Human machine interface for lab abc computer applications','4'],
             ['A survey of user opinion of computer system response time','3'],
             ['The EPS user interface management system','2']]
documents = [sub_list[0] for sub_list in documents]
print(documents)


# In[ ]:


full_year_energy_calc(array5,array4,array_new)


# In[ ]:


len(full_year_energy_calc(array5,array4,array_new))


# In[ ]:


len(full_year_energy_calc(array5,array4,array_new))


# In[ ]:


array6 = []
for i in range(len(array5)):
    array6 =full_year_energy_calc(array5,array4,array_nu[i])


# In[ ]:


index = [1, 2, 3, 4, 5]
for j in product(*(range(d) for d in index)):
    print(j)


# In[ ]:


def full_year_energy_calc(a, b):
    c = []
    i = 0
    j = 0
    while i <len(a):
        c.append(a[i])
        while j < (len(b)-len(a)):
            #print(len(arr2))
            if b[j]>0.0:
                c.append(c[j]-b[j+1])
            else:
                c.append(0.0)
            if (c[j+1] == c[j]):
                c[j+1] = 0.0
            j+=1
        if c[j]<0.0:
            del b[:j-1]
            print(b[j:])
            del c[j-1:]
        i+=1
    return c


# In[ ]:


array4


# In[ ]:


len(array4)


# In[ ]:


array4[0]

full_year_calc(array5,array4)
# In[ ]:


full_year_energy_calc(array5,array4)


# In[ ]:


another_df = pd.read_excel('/home/thabo/uGrid_uGridNet/uGrid/FullYearEnergy.xlsx')


# In[ ]:





# In[ ]:


what_array = another_df.groupby('Day').count().to_numpy()


# In[ ]:


array_nu = np.delete(what_array, 0, 1)


# In[ ]:


array_nu


# In[ ]:


other_df = pd.read_excel('/home/thabo/uGrid_uGridNet/uGrid/FullYearEnergy.xlsx')


# In[ ]:





# In[ ]:


new_stuff = []
for i in range(len(array5)+1):
    new_stuff.append(other_df.query('Day=={}'.format(i)))


# In[ ]:


new_stuff


# In[ ]:


def rev_str(my_str):
    length = len(my_str)
    for i in range(length - 1, -1, -1):
        yield my_str[i]


# For loop to reverse the string
for char in rev_str("hello"):
    print(char)

