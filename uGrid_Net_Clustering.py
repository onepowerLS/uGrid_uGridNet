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
import time
import matplotlib.pyplot as plt
from random import randint

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
# Extend in y direction

#==============================================================================
# Exclusion Mapper using 2nd array instead of increasing list of indexes
def ExclusionMapper1(ExclusionMap_array,reformatScaler,exclusionBuffer,d_EW_between,d_NS_between,width_new,height_new):
    print("in exclusion mapper")
    #Extend the exclusions to include the buffer zone
    bufferArrayWidth_EW = m.ceil(exclusionBuffer/d_EW_between) #reformatted indexes array width 
    bufferArrayHeight_NS = m.ceil(exclusionBuffer/d_NS_between)
    height_og = len(ExclusionMap_array[:,0,0])
    width_og = len(ExclusionMap_array[0,:,0])
    new_exclusions = np.zeros((width_new,height_new))
    indexes = []
    #t0 = time.time()
    
    for i in range((width_og-1),-1,-1):
        k = int((width_og-1-i)/reformatScaler)
        #print(str(i))
        #t1 = time.time()
        #print(str(t1-t0))
        for j in range(0,height_og,1):
            l = int(j/reformatScaler)
            #print(str(i)+" "+str(j))
            if ExclusionMap_array[j,i,0] != 255: #exclusion zones are both grey and black, safe zones are white (255,255,255)
                #set everything within buffer to 1
                if k < bufferArrayWidth_EW:
                    if l < bufferArrayHeight_NS:
                        new_exclusions[0:k+bufferArrayWidth_EW,0:l+bufferArrayHeight_NS] = 1
                    elif l > height_og-bufferArrayHeight_NS:
                        new_exclusions[0:k+bufferArrayWidth_EW,l-bufferArrayHeight_NS:height_new] = 1
                    else:
                        new_exclusions[0:k+bufferArrayWidth_EW,l-bufferArrayHeight_NS:l+bufferArrayHeight_NS] = 1
                elif k > width_new-bufferArrayWidth_EW:
                    if l < bufferArrayHeight_NS:
                        new_exclusions[k-bufferArrayWidth_EW:width_new,0:l+bufferArrayHeight_NS] = 1
                    elif l > height_og-bufferArrayHeight_NS: 
                        new_exclusions[k-bufferArrayWidth_EW:width_new,l-bufferArrayHeight_NS:height_new] = 1
                    else:
                        new_exclusions[k-bufferArrayWidth_EW:width_new,l-bufferArrayHeight_NS:l+bufferArrayHeight_NS] = 1
                else:
                    if l < bufferArrayHeight_NS:
                        new_exclusions[k-bufferArrayWidth_EW:k+bufferArrayWidth_EW,0:l+bufferArrayHeight_NS] = 1
                    elif l > height_og-bufferArrayHeight_NS: 
                        new_exclusions[k-bufferArrayWidth_EW:k+bufferArrayWidth_EW,l-bufferArrayHeight_NS:height_new] = 1
                    else:
                        new_exclusions[k-bufferArrayWidth_EW:k+bufferArrayWidth_EW,l-bufferArrayHeight_NS:l+bufferArrayHeight_NS] = 1
    del ExclusionMap_array
    print("finished remapping")
    new_exclusions = np.flip(new_exclusions)
    
    for o in range(width_new):
        for p in range(height_new):
            if new_exclusions[o,p] == 1:
                indexes.append([o,p])
    indexes = np.array(indexes)
    print("finished new indexing")

    
    #Save new indexes
    index_csv_name = "indexes_reformatted_%s_bufferzone_%s.csv" %(str(reformatScaler),str(exclusionBuffer))
    np.savetxt(index_csv_name,indexes, delimiter=",")
    return indexes
#==============================================================================



#=============================================================================
# Get Distance between array indexes
def DistanceBWindexes(indexesA,indexesB,d_EW_between,d_NS_between):
    if len(indexesA) == 2:
        #This is a single index submission
        A_sqr = m.pow(((indexesA[0]-indexesB[0])*d_EW_between),2)
        B_sqr = m.pow(((indexesA[1]-indexesB[1])*d_EW_between),2)
    else:
        #This is multiple indexes
        A_sqr = np.zeros((len(indexesA), len(indexesB)))
        B_sqr = np.zeros((len(indexesA), len(indexesB)))
        for i in range(len(indexesA)):
            for j in range(len(indexesB)):
                #print(indexesA)
                A_sqr[i,j] = m.pow(((indexesA[i,0]-indexesB[j,0])*d_EW_between),2)
                #print(indexesA[i,0]-indexesB[j,0])
                #print((indexesA[i,0]-indexesB[j,0])*d_EW_between)
                #print(math.pow(((indexesA[i,0]-indexesB[j,0])*d_EW_between),2))
                B_sqr[i,j] = m.pow(((indexesA[i,1]-indexesB[j,1])*d_NS_between),2)
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
def ExtendY(index_x,index_y,index_excl_comp,max_y,range_limit):
    index_y_og = np.copy(index_y)
    for i in range(range_limit):
        #try up one in y
        index_y = index_y_og + i
        if index_y > 0 and index_y < max_y:
            index_pole_comp = index_x + (index_y*0.00001)
            if index_pole_comp in index_excl_comp:
                #if this spot bad try down one in y with that x
                index_y = index_y_og - i
                if index_y > 0 and index_y < max_y:
                    index_pole_comp = index_x + (index_y*0.00001)
                    if index_pole_comp in index_excl_comp:
                        return 9000000, 0
                    else:
                        return index_x, index_y
            else:
                return index_x, index_y
#==============================================================================
        

#==============================================================================
# Find pole location around clustering mean that is not an exclusion spot
def FindNonExclusionSpot(index_x_og, index_y_og, index_excl_comp,range_limit,max_y,max_x):
    for i in range(range_limit):
        #go up and down in y, with current x
        index_y = np.copy(index_y_og)
        index_x = np.copy(index_x_og)
        index_x_new, index_y_new = ExtendY(index_x,index_y,index_excl_comp,max_y,range_limit)
        if index_x_new == 9000000: #no new solution found
            #try go up one in x
            index_x = index_x_og + i
            if index_x > 0 and index_x < max_x:
                index_pole_comp = index_x + (index_y*0.00001)
                if index_pole_comp in index_excl_comp:
                    #go up and down in y, with x+1
                    index_x_new, index_y_new = ExtendY(index_x,index_y,index_excl_comp,max_y,range_limit)
                    if index_x_new == 9000000: #no new solution found
                        #try go down one (from original) from x
                        index_x = index_x_og - i
                        if index_x > 0 and index_x < max_x:
                            index_pole_comp = index_x + (index_y*0.00001)
                            if index_pole_comp in index_excl_comp:
                                #go up and down in y, with x+1
                                index_x_new, index_y_new = ExtendY(index_x,index_y,index_excl_comp,max_y,range_limit)
                                if not index_x_new == 9000000: #no new solution found
                                    #new solution found, otherwise continuing to next iteration of i
                                    return index_x_new, index_y_new
                            else:
                                return index_x, index_y 
                    else:
                        return index_x_new, index_y_new
                else: 
                    return index_x, index_y
        else:
            return index_x_new,index_y_new
        
#==============================================================================

#==============================================================================
# Matching poles and connections by closest distance
def MatchPolesConn(indexes_conn,indexes_poles,d_EW_between,d_NS_between):
    ConnPoles = np.zeros((len(indexes_conn[:,0]),2))      
    DistanceConnPoles = DistanceBWindexes(indexes_conn,indexes_poles,d_EW_between,d_NS_between)
    for i in range(len(ConnPoles[:,0])):
        ConnPoles[i,0] = np.argmin(DistanceConnPoles[i,:])
        ConnPoles[i,1] = np.min(DistanceConnPoles[i,:])
    return ConnPoles
#==============================================================================

#===============================================================================
# Match Connections to Poles Simplified Version
#Instead of trying to find a feasible solution, just create penalties that can be weighted in the optimization
def PolePlacement(PoleConnMax,reformatScaler,num_clusters,exclusionBuffer,range_limit):                   
    
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
        index_csv_name = "indexes_reformatted_%s_bufferzone_%s.csv" %(str(reformatScaler),str(exclusionBuffer))
        indexes_excl = np.loadtxt(index_csv_name, delimiter=",")
    except:
        print("in except loop")
        #quit()
        indexes_excl = ExclusionMapper1(ExclusionMap_array,reformatScaler,exclusionBuffer,d_EW_between,d_NS_between,width,height)

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
    ConnPole_initial, initial_pole_placement = ClusteringConnections(indexes_conn,num_clusters)
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
            index_x_new, index_y_new = FindNonExclusionSpot(initial_pole_placement[i,0], initial_pole_placement[i,1], index_excl_comp,range_limit,height,width)
            if index_x_new == 9000000: #arbitary large number to act as signal
                print("No new pole placement found for pole " +str(i)+", try expanding pole placement region.")
            else:
                #replace pole index location
                indexes_poles[i,:] = [index_x_new, index_y_new]
                print("New pole placed at "+str(i)+" pole")
        
    #Match Connections and Poles  
    ConnPoles = MatchPolesConn(indexes_conn,indexes_poles,d_EW_between,d_NS_between)
    #Find in use poles and create new pole indexes
    indexes_poles_inuse_x = []
    indexes_poles_inuse_y = []
    for i in range(len(indexes_poles)):
        if i in ConnPoles[:,0]:
            indexes_poles_inuse_x.append(indexes_poles[i,0])
            indexes_poles_inuse_y.append(indexes_poles[i,1])
    indexes_poles_in_use_og = np.transpose(np.array([indexes_poles_inuse_x,indexes_poles_inuse_y]))
    #Redetermine match Connections and Poles  
    ConnPoles = MatchPolesConn(indexes_conn,indexes_poles_in_use_og,d_EW_between,d_NS_between)
    #Calculate the wire distances from all the poles and connections
    total_wire_distance = sum(ConnPoles[:,1])
    #Calculate Cost due to poles and distances
    totalCost = PenaltiesToCost(total_wire_distance, num_clusters, ConnPoles,PoleConnMax)
    
    
    #Determine if any poles are too close by testing removing each pole
    Best_pole_indexes = np.copy(indexes_poles_in_use_og)
    Best_totalCost = np.copy(totalCost)
    Best_ConnPoles = np.copy(ConnPoles)
    Best_total_wire_distance = np.copy(total_wire_distance)
    for i in range(len(indexes_poles_in_use_og)):
        indexes_poles_in_use = np.delete(indexes_poles_in_use_og,i,0)
        ConnPoles = MatchPolesConn(indexes_conn,indexes_poles_in_use,d_EW_between,d_NS_between)
        total_wire_distance = sum(ConnPoles[:,1])
        totalCost = PenaltiesToCost(total_wire_distance, num_clusters, ConnPoles,PoleConnMax)
        if totalCost < Best_totalCost:
            Best_pole_indexes = np.copy(indexes_poles_in_use)
            Best_totalCost = np.copy(totalCost)
            Best_ConnPoles = np.copy(ConnPoles)
            Best_total_wire_distance = np.copy(total_wire_distance)
    
    #Calculate Cost due to poles and distances
    totalCost = PenaltiesToCost(total_wire_distance, num_clusters, ConnPoles,PoleConnMax)
    
    return Best_ConnPoles, Best_total_wire_distance, Best_pole_indexes, Best_totalCost, indexes_conn, indexes_excl 
#==============================================================================
             
 
 

#==============================================================================
# Calculate the Cost of the penalties to use as the minimizing optimization value
def PenaltiesToCost(total_wire_distance, num_poles_in_use, ConnPoles,PoleConnMax):
    #Load all Econ input
    Econ_Parameters = pd.read_excel('uGrid_Input.xlsx', sheet_name = 'Econ')

    #Pull out costs needed for penalties
    Cost_Dist_wire = Econ_Parameters['Cost_Dist_wire'][0]
    Cost_Pole = Econ_Parameters['Cost_Pole'][0] + Econ_Parameters['Cost_Pole_Trans'][0]
    #Cost_Dist_Board = Econ_Parameters['Cost_Dist_Board'][0]

    #Calculate pole setup cost
    total_wire_cost = Cost_Dist_wire*((total_wire_distance/1000)**2) #cost is in km, wire distance is in m
    #make total_wire_distance doubly as penalty
    total_pole_cost = num_poles_in_use*Cost_Pole
    #num_dist_boards = 0
    meter_cost_poles= []
    for j in range(num_poles_in_use):
        conn_per_pole = int(np.ceil(np.count_nonzero(ConnPoles[:,0]==j)/PoleConnMax))
        meter_cost = -0.0042*conn_per_pole**5 + 0.1604*conn_per_pole**4 - 2.3536*conn_per_pole**3 + 16.776*conn_per_pole**2 - 59.5*conn_per_pole + 111.25
        meter_cost_poles.append(meter_cost)
    total_smart_meter_cost = sum(meter_cost_poles)
    
    Total_cost = total_wire_cost + total_pole_cost + total_smart_meter_cost
    
    return Total_cost
#==============================================================================        
 

##=============================================================================
# Decide the number of poles and their places by cycling through PolePlacement
def PoleOpt(PoleConnMax,reformatScaler,minPoles,maxPoles,exclusionBuffer,range_limit,MaxDistancePoleConn):
    for i in range(minPoles,maxPoles):
        for j in range(5): #repeat at each cluster level to ensuring repeatability
            t0 = time.time()
            print("Number of poles tried is "+str(i)+".")
            ConnPoles, total_wire_distance, indexes_poles, totalCost, indexes_conn, indexes_excl  = PolePlacement(PoleConnMax,reformatScaler,i,exclusionBuffer,range_limit)
            # Save initial as best solutions
            if i == minPoles:
                ConnPole_soln = np.copy(ConnPoles)
                total_wire_distance_soln = np.copy(total_wire_distance)
                indexes_poles_soln = np.copy(indexes_poles)
                totalCost_soln = np.copy(totalCost)
                num_initial_clusters = np.copy(i)
                #save lowest cost solution as best solution
            else:
                if totalCost < totalCost_soln and max(ConnPoles[:,1] < MaxDistancePoleConn):
                    ConnPole_soln = np.copy(ConnPoles)
                    total_wire_distance_soln = np.copy(total_wire_distance)
                    indexes_poles_soln = np.copy(indexes_poles)
                    totalCost_soln = np.copy(totalCost)
                    num_initial_clusters = np.copy(i)
            t1 = time.time()
            total_time = t1-t0
            print("Time for this pole count is "+str(total_time)+".")
        
    #Save solution pole indexes
    filename = "indexes_poles_reformatted_%s_soln.csv" %(str(reformatScaler))
    np.savetxt(filename,indexes_poles_soln, delimiter=",")
    filename = "ConnPoles_reformatted_%s_soln.csv" %(str(reformatScaler))
    np.savetxt(filename,ConnPole_soln, delimiter=",")
    
    return num_initial_clusters,ConnPole_soln, total_wire_distance_soln, indexes_poles_soln, totalCost_soln, indexes_conn, indexes_excl
#==============================================================================
                
##=============================================================================
# Plot Pole Placement Solution
def PlotPoleSolutions(indexes_poles,indexes_conn,indexes_excl,ConnPoles):
    cmap =  plt.cm.get_cmap('hsv', len(indexes_poles[:,0]))
        
    fig, ax = plt.subplots()
    for i in range(len(indexes_poles[:,0])):
        for j in range(len(ConnPoles[:,0])):
            if ConnPoles[j,0] == i:
                plt.scatter(indexes_conn[j, 0], indexes_conn[j, 1], s=1,c=cmap(i),marker='.')#, color=color)
        plt.scatter(indexes_poles[i,0],indexes_poles[i,1],s=3,c=cmap(i), marker= '^')
    ax.set_aspect('equal')
    plotname = "SolutionPlot.png"
    plt.savefig(plotname)
    plt.show()

    fig, ax = plt.subplots()
    plt.scatter(indexes_excl[:,0],indexes_excl[:,1],s=1,c ='r',marker= 's')
    plt.scatter(indexes_poles[:,0],indexes_poles[:,1], s=1, c='b', marker='s')
    ax.set_aspect('equal')
    plotname = "ExclusionsPlotwSolnPoles.png"
    plt.savefig(plotname)
    plt.show()
    
            

##=============================================================================
   
#==============================================================================
# Check for islands by checking if duplicate rows in onoff matrix with determinate
def CheckForIslands(OnOff): #this will also find rings - need to adjust
    num_poles = len(OnOff[:,0])
    DuplicateList = np.zeros((num_poles,num_poles))

    for i in range(num_poles):
        l = 1 #start at 1, so the first value will be the pole number
        DuplicateList
        for j in range(num_poles):
            if OnOff[i,j] == 1:
                DuplicateList[i,l] = j
                l += 1

    #Check Determinate
    determinate = np.linalg.det(OnOff) #if != 0 then full rank (no duplicate rows)
    
    return determinate
#==============================================================================
    
#==============================================================================
# Check for islands using sparse connected components from scipy
def CheckForIslands2(OnOff):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    graph = csr_matrix(OnOff)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)

    return n_components
#=============================================================================

#==============================================================================
# Internal connected components recursive. This works because undirected
def Check_connections(i,visited,OnOff):
    for j in range(len(OnOff[i,:])):
        if OnOff[i,j] == 1 and visited[j] == 0: #if j in a pole (999 is default for not a pole, and that pole is not visited)
            visited[j] = 1
            Check_connections(j,visited,OnOff)
        #else continue to next pole
    #Check OnOff from other direction
    for k in range(len(OnOff[:,i])):
        if OnOff[k,i] == 1 and visited[k] == 0: #if j in a pole (999 is default for not a pole, and that pole is not visited)
            visited[k] = 1
            Check_connections(k,visited,OnOff)
    
    return visited
#==============================================================================
    
#==============================================================================
# Connected Components Code
def ConnectedComponents(OnOff):
    num_poles = len(OnOff[:,0])
    #Check # connected components
    visited = np.zeros(num_poles)
    conn_comp = 0
    for i in range(num_poles):
        if visited[i] == 0:
            visited[i] = 1
            conn_comp += 1
            visited = Check_connections(i,visited,OnOff)
        #else if visited continue to next pole
    
    return conn_comp, visited #return visited to varify everything has been visited
#==============================================================================
            


#==============================================================================
# Wiring Optimization
def WiringOpt(reformatScaler,nrep,deviation_factor):
    
    t0 = time.time()

    #Load solution pole indexes
    filename = "indexes_poles_reformatted_%s_soln.csv" %(str(reformatScaler))
    indexes_poles = np.loadtxt(filename, delimiter=",")
    filename = "d_between_%s.csv" %str(reformatScaler)
    [d_EW_between,d_NS_between] = np.loadtxt(filename,delimiter=",")
    
    #Calculate Distances between all poles
    DistancesBWPoles = DistanceBWindexes(indexes_poles,indexes_poles,d_EW_between,d_NS_between)
    num_poles = len(indexes_poles[:,0])
    
    #Create on/off matrix with random initial solution
    OnOff = np.random.randint(2, size=(num_poles, num_poles))
    
    #Calculate total distance of random initial solution
    total_distance = 0
    for i in range(num_poles):
        for j in range(num_poles-(i+1)):
            k = i+1+j
            if OnOff[i,k] == 1:
                total_distance = total_distance + DistancesBWPoles[i,k]
                
    #Deploy RRT to determine best OnOff Matrix
    #initialize RRT parameters
    record = np.copy(total_distance) #Total in use BW Pole Distance
    bestsoln = np.copy(OnOff) #Wiring_Pole_Matches
    deviation = deviation_factor*record
    record_records = np.zeros(nrep)
    
    for n in range(nrep):
        print(n)
        t1 = time.time()
        print(t1-t0)
        #Save old solution
        oldsoln = np.copy(OnOff)
        
        #Create new solution, and make sure it doesn't create a duplicate row (islanding within grid)
        #Replace 10 spots at a time
        for p in range(10):
            goodToGo = 0
            while goodToGo == 0:
                #Save old solution for this internal checking loop
                oldsoln_int = np.copy(OnOff)
                choose_1stPole = randint(0,num_poles-1)
                choose_2ndPole = randint(0,num_poles-(choose_1stPole+1)) 
                #switch on/off
                if OnOff[choose_1stPole,choose_2ndPole] == 0:
                    OnOff[choose_1stPole,choose_2ndPole] = 1
                else:
                    OnOff[choose_1stPole,choose_2ndPole] = 0
                components = CheckForIslands2(OnOff)
                if components == 1:
                    goodToGo = 1 #no duplicate rows (islands within network) continue on with solution
                else:
                    OnOff = np.copy(oldsoln_int)
          
        #Evaluate new solution
        tempobj = 0
        for i in range(num_poles):
            for j in range(num_poles-(i+1)):
                k = i+1+j
                if OnOff[i,k] == 1:
                    tempobj = tempobj + DistancesBWPoles[i,k]
        
        if tempobj < record+deviation:
            objnow = np.copy(tempobj)
            if objnow < record:
                record = np.copy(objnow)
                deviation = record*deviation_factor
                bestsoln = np.copy(OnOff)
                print("Best record is (total distance): " + str(record))
        else:
           OnOff = np.copy(oldsoln)
        
        record_records[n] = np.copy(record)
    
    t1 = time.time()
    print("Total opt time is "+str(t1-t0)+".")
    
    #Verify solution matches record
    #Evaluate bestsoln
    obj = 0
    for i in range(num_poles):
        for j in range(num_poles-(i+1)):
            k = i+1+j
            if bestsoln[i,k] == 1:
                obj = obj + DistancesBWPoles[i,k]
    if obj != record:
        print("Verification did not work, need to check code, fun fun")
        
    #Save Results
    records = [record,t1-t0]
    filename = "RRT_Wiring_Records_nrep"+str(nrep)+"_devfactor"+str(deviation_factor)+"_wdevchange.csv"
    np.savetxt(filename,records, delimiter=",")
    filename_records = "RRT_Record_Wiring_OnOff_nrep"+str(nrep)+"_devfactor"+str(deviation_factor)+"_wdevchange.csv"
    np.savetxt(filename_records,bestsoln, delimiter=",")
    filename_records = "RRT_Record_Wiring_RecordRecord_nrep"+str(nrep)+"_devfactor"+str(deviation_factor)+"_wdevchange.csv"
    np.savetxt(filename_records,record_records, delimiter=",")
    filename_records = "RRT_Record_Wiring_DistancesBWPoles_nrep"+str(nrep)+"_devfactor"+str(deviation_factor)+"_wdevchange.csv"
    np.savetxt(filename_records,DistancesBWPoles, delimiter=",")

                
    return record, bestsoln, DistancesBWPoles,record_records      
    
#==============================================================================


#==============================================================================
# Wiring Algorithm
def WiringAlg():
    t0 = time.time()

    #Load solution pole indexes
    filename = "indexes_poles_reformatted_%s_soln.csv" %(str(reformatScaler))
    indexes_poles = np.loadtxt(filename, delimiter=",")
    filename = "d_between_%s.csv" %str(reformatScaler)
    [d_EW_between,d_NS_between] = np.loadtxt(filename,delimiter=",")
    
    #Calculate Distances between all poles
    DistancesBWPoles = DistanceBWindexes(indexes_poles,indexes_poles,d_EW_between,d_NS_between)
    num_poles = len(indexes_poles[:,0])
         
    
    #Sort Distances between poles shortest connections to longest
    DistancesBWPoles_sorted = np.sort(DistancesBWPoles,axis=1) #First value in row will be 0, need to skip first column
    
    #Create Initial Solution
    goodToGo =  0
    num_conn_per_pole = 3 #starting value
    while goodToGo == 0:
        OnOff = np.zeros((num_poles,num_poles)) 
        for i in range(num_poles):
            for j in range(1,num_conn_per_pole): #start at 1 to avoid first column of zeros. Nothing is stopping this from going over num_poles
                ind = np.where(DistancesBWPoles == DistancesBWPoles_sorted[i,j]) #This is going to give two values, need adjust so only one half of matrix is considered
                #Place in use connection
                x = ind[0][0]
                y = ind[1][0]
                OnOff[x,y] = 1
                #make sure connection is also noted from the other pole's side (only will add up one side for total distances though so not double counting)
                OnOff[y,x] = 1
        #Make top/right all zeros of matrix
        for i in range(num_poles):
            for j in range(i,num_poles):
                OnOff[i,j] = 0
        # Check for islands
        components, visited = ConnectedComponents(OnOff) #need to make sure OnOff is being filled in way that there aren't rows of zeros 
        if components == 1:
            goodToGo = 1
        else:
            num_conn_per_pole += 1
    
    #Remove connections, starting with longest working towards shortest, check to make sure not creating islands
    DistancesBWPoles_in_use = DistancesBWPoles_sorted[:,1:num_conn_per_pole] #truncate
    DistancesBWPoles_in_use = DistancesBWPoles_in_use.flatten() #flatten
    DistancesBWPoles_in_use = np.sort(DistancesBWPoles_in_use) #sort
    for k in range(len(DistancesBWPoles_in_use)-1,-1,-1):
        OnOff_temp = np.copy(OnOff)
        ind = np.where(DistancesBWPoles == DistancesBWPoles_in_use[k])
        x = ind[0][0]
        y = ind[1][0]
        OnOff_temp[x,y] = 0
        OnOff_temp[y,x] = 0 #make other matching pair 0 as well
        # Check for islands
        components, visited = ConnectedComponents(OnOff)
        if components == 1: #no islands change OnOff to remove that connection
            OnOff = np.copy(OnOff_temp)
        #If islands don't change OnOff pernament solution

    #Solve for total distance of final OnOff solution
    total_distance = 0
    for i in range(num_poles):
        for j in range(0,i):
            if OnOff[i,j] == 1:
                total_distance = total_distance + DistancesBWPoles[i,j]
                
    t1 = time.time()
    total_time = t1-t0
        
    #Save Results
    records = [total_distance,total_time]
    filename = "Wiring_Records_alg.csv"
    np.savetxt(filename,records, delimiter=",")
    filename_records = "Wiring_OnOff_alg.csv"
    np.savetxt(filename_records,OnOff, delimiter=",")
    filename_records = "Wiring_DistancesBWPoles_alg.csv"
    np.savetxt(filename_records,DistancesBWPoles, delimiter=",")
    
    return total_distance, OnOff, DistancesBWPoles, num_conn_per_pole, total_time
    
#==============================================================================


if __name__ == "__main__":
    
    #Set Inputs for optimizations
    reformatScaler = 5 #parameter to decrease the resolution of image (speeds up processing)
    exclusionBuffer = 2 #meters that poles need to be form exclusions (other poles, exclusions, and connections)
    MaxDistancePoleConn = 50 #(m) the maximum distance allowed for a pole to be from a connection
    PoleConnMax = 20 #maximum number of connections allowed per pole
    minPoles = 30
    maxPoles = 31
    range_limit = 500
    nrep = 100000
    deviation_factor = 0.01
    
    #Run Pole Placement Optimization, output is saved as csv files
    #num_initial_clusters,ConnPole_soln, total_wire_distance_soln, indexes_poles_soln, totalCost_soln, indexes_conn, indexes_excl = PoleOpt(PoleConnMax,reformatScaler,minPoles,maxPoles,exclusionBuffer,range_limit,MaxDistancePoleConn)    
    #PlotPoleSolutions(indexes_poles_soln,indexes_conn,indexes_excl,ConnPole_soln)
    
    #Run Wiring Optimization
    #total_distance, OnOff, DistancesBWPoles,record_records = WiringOpt(reformatScaler,nrep,deviation_factor)
    total_distance, OnOff, DistancesBWPoles, num_conn_per_pole, total_time = WiringAlg()


    