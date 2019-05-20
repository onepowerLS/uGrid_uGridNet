# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:45:05 2019

@author: Phy

Plotting the uGrid Net Results
"""
#https://matplotlib.org/users/image_tutorial.html
#https://matplotlib.org/gallery/images_contours_and_fields/layer_images.html

from pdf2image import convert_from_path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

#==============================================================================
# Load Exclusions Plot
def OriginalExclusionsPlot():
    #Import kml pdf file (of exclusions) and convert to jpg
    pages = convert_from_path('MAK_exclusions.pdf',500)
    for page in pages:
        page.save('MAK_exclusions.jpg','JPEG')

    #Convert JPG to array
    ExclusionMap = Image.open('MAK_exclusions.jpg')
    ExclusionMap_array = np.array(ExclusionMap)
    
    im1 = plt.imshow(ExclusionMap_array)

    plt.show()
    #return im1
#==============================================================================

#==============================================================================
# Exclusions with reformatted size
#def ReformattedExclusionsPlot(reformatScaler):
    
    #Import Exclusion Indexes
    #index_csv_name = "indexes_reformatted_%s.csv" %str(reformatScaler)
    #indexes_excl = np.loadtxt(index_csv_name, delimiter=",")

#==============================================================================
# =============================================================================
# ## Map Connections - scatter
# def ConnectionsPlot(reformatScaler):
#     #Import Connections Indexes
# 
#     index_csv_name = "indexes_conn_reformatted_%s.csv" %str(reformatScaler)
#     indexes_conn = np.loadtxt(index_csv_name, delimiter=",")
#     im2 = plt.scatter(indexes_conn[:,0],indexes_conn[:,1],s=2)
#     plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
#     im2.show()
#     im2.savefig('ConnectionsTest1.png')
#     return im2
# 
# ## Map Poles - scatter
# def PolesPlot():
#     
#     filename_indexpolesinuse = "Indexes_Poles_InUse_wCost_WireSqr_nrep_nrep10000_devfactor0.005.csv" 
#     indexes_poles_in_use = np.loadtxt(filename_indexpolesinuse, delimiter=",")
#     im3 = plt.scatter(indexes_poles_in_use[:,0],indexes_poles_in_use[:,1],s=2)
#     im3.show()
#     return im3
# 
# ## Map Wiring - line plot
# 
# ## Overlay Plots
#     
# ## Main
# if __name__ == "__main__":
#     
#     reformatScaler = 5
#     #OriginalExclusionsPlot()
#     im2 = ConnectionsPlot(reformatScaler)
#     im3 = PolesPlot()
#     plt.show()
# =============================================================================
reformatScaler = 5
index_csv_name = "indexes_conn_reformatted_%s.csv" %str(reformatScaler)
indexes_conn = np.loadtxt(index_csv_name, delimiter=",")
im2 = plt.scatter(indexes_conn[:,0],indexes_conn[:,1],s=2,c ='b')

filename_indexpolesinuse = "Indexes_Poles_InUse_wCost_WireSqr_nrep_nrep10000_devfactor0.005.csv" 
indexes_poles_in_use = np.loadtxt(filename_indexpolesinuse, delimiter=",")
im3 = plt.scatter(indexes_poles_in_use[:,0],indexes_poles_in_use[:,1],s=2,c='r')

plt.show()
