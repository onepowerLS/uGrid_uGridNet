# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 09:20:26 2019

uGrid Net
The goal is to take the kml file which has highlighted polygons of "can't build here"
areas and also an inputted generation station gps location, and determine where
to place distribution poles and how to connect the distribution network together.

@author: Phy
"""

import numpy as np
import pandas as pd
import math as m
from pdf2image import convert_from_path
from PIL import Image

#==============================================================================
#Calculate Distance between GPS coordinates with Haversine Formula 
def GPStoDistance(Lat1,Lat2,Long1,Long2):
    R_earth = 6371 #earth's mean radius in km
    a = m.sin((Lat1-Lat2)/2)**2 + m.cos(Lat1)*m.cos(Lat2)*m.sin((Long1-Long2)/2)**2
    c = 2*m.atan2(m.sqrt(a),m.sqrt(1-a))
    d = R_earth*c
    return d
#===============================================================================
    
#==============================================================================
# Create the exclusion arrary: captures the number of exclusions and the indexes of those exclusions
def ExclusionMapper(ExclusionMap_array):
    number_exclusions = 0
    y_index = []
    x_index = []
    for i in range(len(ExclusionMap_array[:,0])):
        print(i)
        for j in range(len(ExclusionMap_array[0,:])):
            if ExclusionMap_array[i,j,0] != 255: #exclusion zones are both grey and black, safe zones are white (255,255,255)
                number_exclusions += 1
                #add saving index location
                y_index.append(i)
                x_index.append(j)
    indexes = np.transpose(np.array([x_index,y_index]))
    np.savetxt("indexes.csv",indexes, delimiter=",")
    return number_exclusions, indexes
#==============================================================================

# Run Full Code ===============================================================
if __name__ == "__main__":
    #Import csv file which has been converted from the klm file
    #This gives the points of connections which are houses to link to the distribution grid
    Connect_nodes = pd.read_excel('MAK_connections.xlsx', sheet_name = 'connections')
    Exclusion_nodes = pd.read_excel('MAK_exclusions.xlsx', sheet_name = 'MAK_exclusions')

    #Identify gps coordinate min and max to determine coordinates of edges of jpg image
    Longitude_exc = Exclusion_nodes['X']
    Latitude_exc = Exclusion_nodes['Y']
    #also convert these degrees to radians
    Lat_exc_min = m.radians(Latitude_exc.min()) #top of image (north)
    Lat_exc_max = m.radians(Latitude_exc.max()) #bottom of image (south)
    Long_exc_min = m.radians(Longitude_exc.min()) #left of image (east)
    Long_exc_max = m.radians(Longitude_exc.max()) #right of image (west)

    #Calculate the distance between the gps coordiantes using Haversine Formula
    #North South Distance #measuring latitude difference
    d_NS = GPStoDistance(Lat_exc_max,Lat_exc_min,Long_exc_max,Long_exc_max) #km
    #East West Distance #measuring longitude difference
    d_EW = GPStoDistance(Lat_exc_max,Lat_exc_max,Long_exc_max,Long_exc_min) #km

    #Import kml pdf file (of exclusions) and convert to jpg
    pages = convert_from_path('MAK_exclusions.pdf',500)
    for page in pages:
        page.save('MAK_exclusions.jpg','JPEG')
    
    #Convert JPG to array
    ExclusionMap = Image.open('MAK_exclusions.jpg')
    ExclusionMap_array = np.array(ExclusionMap)
    #Filter rgb value to 0 'non exclusion' and 1 'exclusion'
    #Black 0-0-0, White 255-255-255
    height = len(ExclusionMap_array[:,0])
    width = len(ExclusionMap_array[0,:])

    #Determine distance between pixels (between values in the array)
    d_EW_between = d_EW/width*1000 #m
    d_NS_between = d_NS/height*1000 #m

    #Load exlusion map, if not available then perform
    try:
        indexes = np.loadtxt("indexes.csv", delimiter=",")
    except: 
        number_exclusions, x_index, y_index = ExclusionMapper(ExclusionMap_array)

    #Match the connection locations to locations in the array
    #Find distance between east limit of image and connection
    d_Econnection = np.zeros(len(Connect_nodes))
    d_Nconnection = np.zeros(len(Connect_nodes))
    for i in range(len(Connect_nodes)): #iteration through connections
        d_Econnection[i] = GPStoDistance(Lat_exc_min,Lat_exc_min,Long_exc_min,Connect_nodes['longitude'][i])
        d_Nconnection[i] = GPStoDistance(Lat_exc_min,Connect_nodes['latitude'][i],Long_exc_min,Long_exc_min)
    #Find the array indexes that match the connection locations

    

    #Setup pole location optimization
    #Check uGrid Net rules 
    #constraints: 
    # 1) can't be in exclusion zone
    # 2) minimum and maximum number of connections in proximity to pole
    # 3) minimum and maximum distnace of pole from connection
    #objective: fewest number of poles
    
    #Random selection of pole locations
    #constraint: cant be same location as exclusion, connection, or previously placed pole


    