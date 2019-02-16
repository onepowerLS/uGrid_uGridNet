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
import math
import pandas as pd
import math as m
from pdf2image import convert_from_path
from PIL import Image
import time
import matplotlib.pyplot as plt

#==============================================================================
#Calculate Distance between GPS coordinates with Haversine Formula (returns in m)
def GPStoDistance(Lat1,Lat2,Long1,Long2):
    R_earth = 6371 #earth's mean radius in km
    a = m.sin((Lat1-Lat2)/2)**2 + m.cos(Lat1)*m.cos(Lat2)*m.sin((Long1-Long2)/2)**2
    c = 2*m.atan2(m.sqrt(a),m.sqrt(1-a))
    d = (R_earth*c)*1000 #m
    return d
#===============================================================================
    
#==============================================================================
# Create the exclusion arrary: captures the number of exclusions and the indexes of those exclusions
# This adjusts the indexes of the original picture so it is in the same format as the connections
def ExclusionMapper(ExclusionMap_array,reformatScaler):
    y_index = []
    x_index = []
    i = len(ExclusionMap_array[:,0])-1
    k = 0
    while i >=0:
        j=0
        while j < len(ExclusionMap_array[0,:]):
            if ExclusionMap_array[i,j,0] != 255: #exclusion zones are both grey and black, safe zones are white (255,255,255)
                #Reformate Exclusion map to decreased resolution
                #Current resolution is to .15 m. Decreasing resolution to 1.5 m.
                #Current plan is to cut off bottom and right side of picture up to 1.5m (10 pixels)
                #if there are any exclusions in the new resolution square the whole square is marked as exclusion 
                new_x = int(k/reformatScaler)
                new_y = int(j/reformatScaler)
                #add saving index location
                y_index.append(new_y)
                x_index.append(new_x)
            j+=1
        i -= 1
        k+=1
    #indexes = np.array([x_index,y_index])
    indexes = np.transpose(np.array([y_index,x_index]))
    #np.flip(indexes,1)
    index_csv_name = "indexes_reformatted_%s.csv" %str(reformatScaler)
    np.savetxt(index_csv_name,indexes, delimiter=",")
    return  indexes
#==============================================================================


#=============================================================================
# Get Distance between array indexes
def DistanceBWindexes(indexesA,indexesB,d_EW_between,d_NS_between):
    if len(indexesA) == 2:
        #This is a single index submission
        A_sqr = math.pow(((indexesA[0]-indexesB[0])*d_EW_between),2)
        B_sqr = math.pow(((indexesA[1]-indexesB[1])*d_EW_between),2)
    else:
        #This is multiple indexes
        A_sqr = np.zeros((len(indexesA), len(indexesB)))
        B_sqr = np.zeros((len(indexesA), len(indexesB)))
        for i in range(len(indexesA)):
            for j in range(len(indexesB)):
                #print(indexesA)
                A_sqr[i,j] = math.pow(((indexesA[i,0]-indexesB[j,0])*d_EW_between),2)
                #print(indexesA[i,0]-indexesB[j,0])
                #print((indexesA[i,0]-indexesB[j,0])*d_EW_between)
                #print(math.pow(((indexesA[i,0]-indexesB[j,0])*d_EW_between),2))
                B_sqr[i,j] = math.pow(((indexesA[i,1]-indexesB[j,1])*d_NS_between),2)
    DistanceAB = np.sqrt(A_sqr+B_sqr)
    return DistanceAB
#=============================================================================

#==============================================================================
# Clustering of Connections for Initial Solution Pole Placement
def ClusteringConnections(indexes_conn,num_clusters):
    from sklearn import mixture
    
    X = np.copy(indexes_conn)

    cv_type = 'tied'
    gmm = mixture.GaussianMixture(n_components = num_clusters,covariance_type=cv_type)
    gmm.fit(X)
    clf = gmm 
    Y_ = clf.predict(X) #Y_ the index is the connection # and the value is the pole
    #gmm.means_: the index is the pole #, the numbers are the indexes of that pole
    
    #round and convert means to integers
    means = np.copy(gmm.means_)
    for i in range(len(gmm.means_)):
        means[i][0] = int(means[i][0])
        means[i][1] = int(means[i][1])
    
    return Y_ ,means
#==============================================================================

#==============================================================================
# Exclusion testing around y
def ExtendY(index_x,index_y,index_excl_comp,i):
    index_y_og = np.copy(index_y)    
    #try up one in y
    index_y = index_y_og + i
    index_pole_comp = index_x + (index_y*0.00001)
    if index_pole_comp in index_excl_comp:
        #if this spot bad try down one in y with that x
        index_y = index_y_og - i
        index_pole_comp = index_x + (index_y*0.00001)
        if index_pole_comp in index_excl_comp:
            return 0, 0
        else:
            return index_x, index_y
    else:
        return index_x, index_y
#==============================================================================
        

#==============================================================================
# Find pole location around clustering mean that is not an exclusion spot
def FindNonExclusionSpot(index_x_og, index_y_og, index_excl_comp):
    for i in range(20):
        #go up and down in y, with current x
        index_y = np.copy(index_y_og)
        index_x = np.copy(index_x_og)
        index_x_new, index_y_new = ExtendY(index_x,index_y,index_excl_comp,i)
        if index_x_new == 0: #no new solution found
            #try go up one in x
            index_x = index_x_og + i
            index_pole_comp = index_x + (index_y*0.00001)
            if index_pole_comp in index_excl_comp:
                #go up and down in y, with x+1
                index_x_new, index_y_new = ExtendY(index_x,index_y,index_excl_comp,i)
                if index_x_new == 0: #no new solution found
                    #try go down one (from original) from x
                    index_x = index_x_og - i
                    index_pole_comp = index_x + (index_y*0.00001)
                    if index_pole_comp in index_excl_comp:
                        #go up and down in y, with x+1
                        index_x_new, index_y_new = ExtendY(index_x,index_y,index_excl_comp,i)
                        if index_x_new == 0: #no new solution found
                            #no new solution found, go to next iteration of i
                            goodToGo = 0 #this is placehold value so if function can run
                        else:
                            return index_x_new, index_y_new
                    else:
                        return index_x, index_y 
                else:
                    return index_x_new, index_y_new
            else: 
                return index_x, index_y
        else: 
            return index_x_new, index_y_new
    else: 
        return index_x, index_y
        
    return 0,0
#==============================================================================

#===============================================================================
# Match Connections to Poles Simplified Version
#Instead of trying to find a feasible solution, just create penalties that can be weighted in the optimization
def PolePlacement(PoleConnMax,reformatScaler,num_clusters):                   
    
    #Gather the information needed
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
    d_NS = GPStoDistance(Lat_exc_max,Lat_exc_min,Long_exc_max,Long_exc_max) #m
    #East West Distance #measuring longitude difference
    d_EW = GPStoDistance(Lat_exc_max,Lat_exc_max,Long_exc_max,Long_exc_min) #m

    #Import kml pdf file (of exclusions) and convert to jpg
    pages = convert_from_path('MAK_exclusions.pdf',500)
    for page in pages:
        page.save('MAK_exclusions.jpg','JPEG')
    
    #Convert JPG to array
    ExclusionMap = Image.open('MAK_exclusions.jpg')
    ExclusionMap_array = np.array(ExclusionMap)
    #Filter rgb value to 0 'non exclusion' and 1 'exclusion'
    #Black 0-0-0, White 255-255-255
    height = int(len(ExclusionMap_array[:,0])/reformatScaler) #this is y_index_max
    width = int(len(ExclusionMap_array[0,:])/reformatScaler) #this is x_index_max
    filename = "index_maxes_%s.csv" %str(reformatScaler)
    np.savetxt(filename,[height,width],delimiter=",")

    #Determine distance between reformatted pixels (between values in the array)
    d_EW_between = d_EW/width #m
    d_NS_between = d_NS/height #m
    filename = "d_between_%s.csv" %str(reformatScaler)
    np.savetxt(filename,[d_EW_between,d_NS_between],delimiter=",")

    #Load exlusion map, if not available then perform
    #This gathers the exclusion array indexes
    try:
        #print("in try loop")
        index_csv_name = "indexes_reformatted_%s.csv" %str(reformatScaler)
        indexes_excl = np.loadtxt(index_csv_name, delimiter=",")
    except:
        #print("in except loop")
        #quit()
        indexes_excl = ExclusionMapper(ExclusionMap_array,reformatScaler)

    #Match the connection locations to locations in the array
    #Find distance between east limit of image and connection
    try:
        #print("in try loop")
        index_csv_name = "indexes_conn_reformatted_%s.csv" %str(reformatScaler)
        indexes_conn = np.loadtxt(index_csv_name, delimiter=",")
    except:
        d_Econnection = np.zeros(len(Connect_nodes))
        d_Nconnection = np.zeros(len(Connect_nodes))
        indexes_conn = np.zeros((len(Connect_nodes),2))
        for i in range(len(Connect_nodes)): #iteration through connections
            d_Econnection[i] = GPStoDistance(Lat_exc_min,Lat_exc_min,Long_exc_min,m.radians(Connect_nodes['longitude'][i])) #m
            #print(d_Econnection[i])
            #distance of connection to the east (left) (x index)
            d_Nconnection[i] = GPStoDistance(Lat_exc_min,m.radians(Connect_nodes['latitude'][i]),Long_exc_min,Long_exc_min) #m
            #print(d_Nconnection[i])
            #distance of connection to the north (top) (y index)
            #Get array index locations of all connections
            indexes_conn[i,0] = int(d_Econnection[i]/d_EW_between)
            #print(indexes_conn[i,0])
            indexes_conn[i,1] = int(d_Nconnection[i]/d_NS_between)
            #print(indexes_conn[i,1])
            index_csv_name = "indexes_conn_reformatted_%s.csv" %str(reformatScaler)
            np.savetxt(index_csv_name,indexes_conn, delimiter=",")
    
    #Get initial solution of pole indexes with Gaussian Mean Clustering
    #The mean of the clusters is the initial pole placement
    ConnPole, initial_pole_placement = ClusteringConnections(indexes_conn,num_clusters)
    indexes_poles = np.copy(initial_pole_placement)
    #ConnPole: the index is the connection # and the value is the pole
    #initial_pole_placement: the index is the pole #, the numbers are the indexes of that pole

    #Check if pole placed on exclusion
    #If in exclusion zone, circle outwardly to find non-exclusion spot
    #Turn 2D index arrays to 1D for comparison
    index_pole_comp = initial_pole_placement[:,0] + np.dot(initial_pole_placement[:,1],0.00001)
    index_excl_comp = indexes_excl[:,0] + np.dot(indexes_excl[:,1],0.00001)
    for i in range(len(index_pole_comp)):
        if index_pole_comp[i] in index_excl_comp:
            #check surrounding areas for non-exclusion spot
            index_x_new, index_y_new = FindNonExclusionSpot(initial_pole_placement[i,0], initial_pole_placement[i,1], index_excl_comp)
            if index_x_new == 0:
                print("No pole placement found need to try cluster again, or expand region tested.")
            else:
                #replace pole index location
                indexes_poles[i,:] = [index_x_new, index_y_new]
                print("New pole placed at "+str(i)+" pole")
        
    #Calculate Distances between poles and connections 
    Dist_ConnPole = np.zeros(len(ConnPole))
    for k in range(len(ConnPole)):
        Dist_ConnPole[k] = DistanceBWindexes(indexes_conn[k,:],indexes_poles[ConnPole[k],:],d_EW_between,d_NS_between)

    #Calculate the wire distances from all the poles and connections
    total_wire_distance = sum(Dist_ConnPole)
    
    #Calculate Cost due to poles and distances
    totalCost = PenaltiesToCost(total_wire_distance, num_clusters, ConnPole,PoleConnMax)
    
    return ConnPole, total_wire_distance, indexes_poles, Dist_ConnPole,totalCost, indexes_conn, indexes_excl 
#==============================================================================
             
 
 

#==============================================================================
# Calculate the Cost of the penalties to use as the minimizing optimization value
def PenaltiesToCost(total_wire_distance, num_poles_in_use, ConnPoles,PoleConnMax):
    #Load all Econ input
    Econ_Parameters = pd.read_excel('uGrid_Input.xlsx', sheet_name = 'Econ')

    #Pull out costs needed for penalties
    Cost_Dist_wire = Econ_Parameters['Cost_Dist_wire'][0]
    Cost_Pole = Econ_Parameters['Cost_Pole'][0] + Econ_Parameters['Cost_Pole_Trans'][0]
    Cost_Dist_Board = Econ_Parameters['Cost_Dist_Board'][0]

    #Calculate pole setup cost
    total_wire_cost = Cost_Dist_wire*((total_wire_distance/1000)**2) #cost is in km, wire distance is in m
    #make total_wire_distance doubly as penalty
    total_pole_cost = num_poles_in_use*Cost_Pole
    num_dist_boards = 0
    for j in range(num_poles_in_use):
        board_per_pole = int(np.ceil(np.count_nonzero(ConnPoles==j)/PoleConnMax))
        num_dist_boards = num_dist_boards + board_per_pole
    total_distboard_cost = num_dist_boards*Cost_Dist_Board
    
    Total_cost = total_wire_cost + total_pole_cost + total_distboard_cost
    
    return Total_cost
#==============================================================================        
 

##=============================================================================
# Decide the number of poles and their places by cycling through PolePlacement
def PoleOpt(PoleConnMax,reformatScaler,minPoles,maxPoles):
    for i in range(minPoles,maxPoles):
        t0 = time.time()
        print("Number of poles tried is "+str(i)+".")
        ConnPole, total_wire_distance, indexes_poles, Dist_ConnPole,totalCost, indexes_conn, indexes_excl  = PolePlacement(PoleConnMax,reformatScaler,i)
        # Save best solutions
        if i == minPoles:
            ConnPole_soln = np.copy(ConnPole)
            total_wire_distance_soln = np.copy(total_wire_distance)
            indexes_poles_soln = np.copy(indexes_poles)
            Dist_ConnPole_soln = np.copy(Dist_ConnPole)
            totalCost_soln = np.copy(totalCost)
        else:
            if totalCost < totalCost_soln:
                ConnPole_soln = np.copy(ConnPole)
                total_wire_distance_soln = np.copy(total_wire_distance)
                indexes_poles_soln = np.copy(indexes_poles)
                Dist_ConnPole_soln = np.copy(Dist_ConnPole_soln)
                totalCost_soln = np.copy(totalCost_soln)
        t1 = time.time()
        total_time = t1-t0
        print("Time for this pole count is "+str(total_time)+".")
        
        #Save solution pole indexes
        filename = "indexes_poles_reformatted_%s_soln_%s.csv" %(str(reformatScaler),str(len(indexes_poles_soln[:,0])))
        np.savetxt(filename,indexes_poles_soln, delimiter=",")
    
    return ConnPole_soln, total_wire_distance_soln, indexes_poles_soln, Dist_ConnPole_soln, totalCost_soln, indexes_conn, indexes_excl
#==============================================================================
                
##=============================================================================
# Plot Pole Placement Solution
def PlotPoleSolutions(indexes_poles,indexes_conn,indexes_excl,ConnPole):
    cmap =  plt.cm.get_cmap('hsv', len(indexes_poles[:,0]))
        
    fig, ax = plt.subplots()
    for i in range(max(ConnPole)+1):
        if i in ConnPole:
            plt.scatter(indexes_conn[ConnPole == i, 0], indexes_conn[ConnPole == i, 1], s=1,c=cmap(i),marker='.')#, color=color)
            plt.scatter(indexes_poles[i,0],indexes_poles[i,1],s=3,c=cmap(i), marker= '^')
    ax.set_aspect('equal')
    plotname = "SolutionPlot.png"
    plt.savefig(plotname)
    plt.show()

    fig, ax = plt.subplots()
    plt.scatter(indexes_excl[:,0],indexes_excl[:,1],s=1,c ='r',marker= 's')
    ax.set_aspect('equal')
    plotname = "ExclusionsPlot.png"
    plt.savefig(plotname)
    plt.show()
    
            

##=============================================================================
   


if __name__ == "__main__":
    
    #Set Inputs for optimizations
    reformatScaler = 5 #parameter to decrease the resolution of image
    #exclusion_buffer = 20 #meters that poles need to be form exclusions (other poles, exclusions, and connections)
    #MaxDistancePoleConn = 50 #(m) the maximum distance allowed for a pole to be from a connection
    PoleConnMax = 20 #maximum number of connections allowed per pole
    minPoles = 70
    maxPoles = 80
    
    #Run Pole Placement Optimization, output is saved as csv files
    ConnPole_soln, total_wire_distance_soln, indexes_poles_soln, Dist_ConnPole_soln, totalCost_soln, indexes_conn, indexes_excl = PoleOpt(PoleConnMax,reformatScaler,minPoles,maxPoles)    
    PlotPoleSolutions(indexes_poles_soln,indexes_conn,indexes_excl,ConnPole_soln)



    