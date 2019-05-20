# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:25:14 2019

@author: Phy
"""

"""
Created on Thu Jan 10 09:20:26 2019

uGrid Net
The goal is to take the kml file which has highlighted polygons of "can't build here"
areas and also an inputted generation station gps location, and determine where
to place distribution poles and how to connect the distribution network together.

Including Reliability Cost-Benefit into layout optimization

@author: Phy
"""

import numpy as np
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
    if len(indexesA) == 2 and len(indexesB) == 2:
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
# Find non-exlcusion spot in growing circular manner
def FindNonExclusionSpot(index_x_og, index_y_og, index_excl_comp,range_limit,max_y,max_x):
    for i in range(range_limit):
        #Define limits
        x_min = int(index_x_og - i)
        if x_min < 0:
            x_min = 0
        x_max = int(index_x_og + i)
        if x_max > max_x:
            x_max = max_x
        y_min = int(index_y_og - i)
        if y_min < 0:
            y_min = 0
        y_max = int(index_y_og + i)
        if y_max > max_y:
            y_max = 0
        #Testing around spot
        #xmin and xmax
        for y in range(y_min,y_max):
            index_pole_comp = x_min + (y*0.00001)
            if not index_pole_comp in index_excl_comp:
                return x_min, y
            index_pole_comp = x_max + (y*0.00001)
            if not index_pole_comp in index_excl_comp:
                return x_min, y
        #ymin and ymax
        for x in range(x_min,x_max):
            index_pole_comp = x + (y_min*0.00001)
            if not index_pole_comp in index_excl_comp:
                return x, y_min
            index_pole_comp = x + (y_max*0.00001)
            if not index_pole_comp in index_excl_comp:
                return x, y_max
    
    #if nothing found in within range limit, return dummy number
    return 9000000,0
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
def PolePlacement(reformatScaler,num_clusters,exclusionBuffer,range_limit,indexes_conn,indexes_excl,height,width,d_EW_between,d_NS_between,prob, Cost_kWh, wiring_trans_cost, restoration_time,Long_exc_min,Lat_exc_min):                    
      
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
                #print("New pole placed at "+str(i)+" pole")
        
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
    total_distance_BWPoles,Best_Wire_Cost,Best_Reliability_Cost,Best_Total_Cost, OnOff, DistancesBWPoles, num_conn_per_pole, total_time = WiringAlg(ConnPoles,prob, Cost_kWh, wiring_trans_cost, restoration_time,Long_exc_min,Lat_exc_min,indexes_poles,d_EW_between,d_NS_between,"False")
    totalCost_minusReliability = PenaltiesToCost(total_wire_distance, num_clusters, ConnPoles,total_distance_BWPoles)
    totalCost = totalCost_minusReliability + Best_Reliability_Cost
    
    #Determine if any poles are too close by testing removing each pole
    #This takes too long with reliability
    #print("Testing pole removal")
    Best_pole_indexes = np.copy(indexes_poles_in_use_og)
    Best_totalCost = np.copy(totalCost)
    Best_ConnPoles = np.copy(ConnPoles)
    Best_total_wire_distance = np.copy(total_wire_distance)
    Best_total_distance_BWPoles = np.copy(total_distance_BWPoles)
    Best_OnOff = np.copy(OnOff)
    #for i in range(len(indexes_poles_in_use_og)):
    #    indexes_poles_in_use = np.delete(indexes_poles_in_use_og,i,0)
    #    ConnPoles = MatchPolesConn(indexes_conn,indexes_poles_in_use,d_EW_between,d_NS_between)
    #    total_wire_distance = sum(ConnPoles[:,1])
    #    total_distance_BWPoles,Best_Wire_Cost,Best_Reliability_Cost,Best_Total_Cost, OnOff, DistancesBWPoles, num_conn_per_pole, total_time = WiringAlg(ConnPoles,prob, Cost_kWh, wiring_trans_cost, restoration_time,Long_exc_min,Lat_exc_min,indexes_poles,d_EW_between,d_NS_between,"False")
    #    totalCost_minusReliability = PenaltiesToCost(total_wire_distance, num_clusters, ConnPoles,total_distance_BWPoles)
    #    totalCost = totalCost_minusReliability + Best_Reliability_Cost
    #    if totalCost < Best_totalCost:
    #        print("Removing Pole, cheaper without")
    #        Best_pole_indexes = np.copy(indexes_poles_in_use)
    #        Best_totalCost = np.copy(totalCost)
    #        Best_ConnPoles = np.copy(ConnPoles)
    #        Best_total_wire_distance = np.copy(total_wire_distance)
    #        Best_total_distance_BWPoles = np.copy(total_distance_BWPoles)
    #        Best_OnOff = np.copy(OnOff)
    #print("Best total Cost and number of cluster")
    #print(Best_totalCost)
    #print(num_clusters)
    #print("******************")
       
    return Best_Reliability_Cost, Best_OnOff, Best_ConnPoles, Best_total_wire_distance, Best_total_distance_BWPoles, Best_pole_indexes, Best_totalCost

#==============================================================================
             
 
 

#==============================================================================
# Calculate the Cost of the penalties to use as the minimizing optimization value
def PenaltiesToCost(total_wire_distance, num_poles_in_use, ConnPoles,total_distance_BWPoles):
    #Load all Econ input
    Econ_Parameters = pd.read_excel('uGrid_Input.xlsx', sheet_name = 'Econ')

    #Pull out costs needed for penalties
    Cost_Dist_wire = Econ_Parameters['Cost_Dist_wire'][0]
    Cost_Trans_wire = Econ_Parameters['Cost_Trans_wire'][0]
    Cost_Pole = Econ_Parameters['Cost_Pole'][0] + Econ_Parameters['Cost_Pole_Trans'][0]
    #Cost_Dist_Board = Econ_Parameters['Cost_Dist_Board'][0]

    #Calculate pole setup cost
    total_dist_wire_cost = Cost_Dist_wire*(total_wire_distance/1000) #cost is in km, wire distance is in m
    total_trans_wire_cost = Cost_Trans_wire*(total_distance_BWPoles/1000)
    #make total_wire_distance doubly as penalty
    total_pole_cost = num_poles_in_use*Cost_Pole
    #num_dist_boards = 0
    meter_cost_poles= []
    for j in range(num_poles_in_use):
        conn_per_pole = int(np.ceil(np.count_nonzero(ConnPoles[:,0]==j)))
        meter_cost = -0.0042*conn_per_pole**5 + 0.1604*conn_per_pole**4 - 2.3536*conn_per_pole**3 + 16.776*conn_per_pole**2 - 59.5*conn_per_pole + 111.25
        meter_cost_poles.append(meter_cost)
    total_smart_meter_cost = sum(meter_cost_poles)
    
    Total_cost = total_dist_wire_cost + total_trans_wire_cost + total_pole_cost + total_smart_meter_cost
    
    return Total_cost
#==============================================================================        
 

##=============================================================================
# Decide the number of poles and their places by cycling through PolePlacement
def PoleOpt(reformatScaler,minPoles,maxPoles,exclusionBuffer,range_limit,MaxDistancePoleConn,repeats,prob, Cost_kWh, wiring_trans_cost, restoration_time):
    
    #Load Files
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
    
    #Find Poles, Wiring, and Cost
    lookback = 3 * repeats
    record_bests = np.zeros((maxPoles-minPoles)*repeats)
    p = 0
    i = minPoles
    NoDecrease = 0
    totalCost_soln = 9999999999999 #initialize with very high totalCost_soln
    while i < maxPoles and NoDecrease == 0:
        for j in range(repeats): #repeat at each cluster level to ensuring repeatability
            t0 = time.time()
            print("Number of poles tried is "+str(i)+" with "+str(j+1)+" out of "+str(repeats)+" attempts.")                   
    
            Reliability_Cost, OnOff, ConnPoles, total_dist_wire_distance, total_trans_wire_distance, indexes_poles, totalCost  = PolePlacement(reformatScaler,i,exclusionBuffer,range_limit,indexes_conn,indexes_excl,height,width,d_EW_between,d_NS_between,prob, Cost_kWh, wiring_trans_cost, restoration_time,Long_exc_min,Lat_exc_min)
            if totalCost < totalCost_soln and max(ConnPoles[:,1]) < MaxDistancePoleConn:
                ConnPole_soln = np.copy(ConnPoles)
                total_dist_wire_distance_soln = np.copy(total_dist_wire_distance)
                total_trans_wire_distance_soln = np.copy(total_trans_wire_distance)
                indexes_poles_soln = np.copy(indexes_poles)
                totalCost_soln = np.copy(totalCost)
                num_initial_clusters = np.copy(i)
                OnOff_soln = np.copy(OnOff)
                Best_Reliability_Cost = np.copy(Reliability_Cost)
            print("Best total cost for this iteration is $"+str(int(totalCost))+", and the current best solution is $"+str(int(totalCost_soln))+".")
            print("The maximum distance between house and pole is "+str(int(max(ConnPoles[:,1])))+"m, and limit is "+str(MaxDistancePoleConn)+"m.")
            record_bests[p] = totalCost
            p += 1
            t1 = time.time()
            total_time = t1-t0
            print("Time for this pole count is "+str(round(total_time, 2))+".")
            #Check if the solution has not gotten better for the last 3 pole # increases, if so solution is found and exit this loop
            if p > lookback:
                if min(record_bests[p-lookback:p]) > totalCost_soln:
                    NoDecrease = 1            
        i += 1
                    
        
    #Save solution pole indexes
    filename = "indexes_poles_reformatted_%s_prob_%s_FullReliability_soln.csv" %(str(reformatScaler),str(prob))
    np.savetxt(filename,indexes_poles_soln, delimiter=",")
    filename = "ConnPoles_reformatted_%s_prob_%s_FullReliability_soln.csv" %(str(reformatScaler),str(prob))
    np.savetxt(filename,ConnPole_soln, delimiter=",")
    filename_records = "Wiring_OnOff_alg_prob_%s_FullReliability.csv" %(str(prob))
    np.savetxt(filename_records,OnOff_soln, delimiter=",")
    
    return Best_Reliability_Cost, OnOff_soln,num_initial_clusters,ConnPole_soln, total_dist_wire_distance_soln, total_trans_wire_distance_soln, indexes_poles_soln, totalCost_soln, indexes_conn, indexes_excl
#==============================================================================
    
#==============================================================================
# Plot between pole wiring
def PoleWiring(OnOff, indexes_poles):

    num_poles = len(indexes_poles[:,0])
    goodToGo = 0
    for i in range(num_poles):
        for j in range(i):
            if OnOff[i,j] == 1:
                match = np.array([[indexes_poles[i,0],indexes_poles[i,1]],[indexes_poles[j,0],indexes_poles[j,1]]]) 
                if goodToGo == 0:
                    goodToGo = 1
                    wiringMatrix = match
                else:
                    wiringMatrix = np.concatenate((wiringMatrix,match),axis=1)
    
    #Plot
    num_match = len(wiringMatrix[0,:])
    fig, ax = plt.subplots()
    for i in range(0,num_match-1,2):
        j = i + 1
        plt.plot(wiringMatrix[:,i],wiringMatrix[:,j])
    #plt.scatter(indexes_poles[:,0],indexes_poles[:,1],s=2,c='b')
    ax.set_aspect('equal')
    plotname = "WiringBWPoles_prob_%s_FullReliability.png" %(str(prob))
    plt.savefig(plotname, dpi=600)
    plt.show()
#=============================================================================
          
#==============================================================================
# plot all wiring
def AllWiringPlot(OnOff, indexes_poles, indexes_conn, ConnPoles):
    
    #Pole Wiring
    num_poles = len(indexes_poles[:,0])
    goodToGo = 0
    for i in range(num_poles):
        for j in range(i):
            if OnOff[i,j] == 1:
                match = np.array([[indexes_poles[i,0],indexes_poles[i,1]],[indexes_poles[j,0],indexes_poles[j,1]]]) 
                if goodToGo == 0:
                    goodToGo = 1
                    wiringMatrix = match
                else:
                    wiringMatrix = np.concatenate((wiringMatrix,match),axis=1)
    
    #Plot
    num_match = len(wiringMatrix[0,:])
    fig, ax = plt.subplots()
    for i in range(0,num_match-1,2):
        j = i + 1
        plt.plot(wiringMatrix[:,i],wiringMatrix[:,j],color='black',linewidth=2)
    
    #Connections wiring
    num_conn = len(indexes_conn[:,0])
    for i in range(num_conn):
        pole = int(ConnPoles[i,0])
        x_s = [indexes_conn[i,0],indexes_poles[pole,0]]
        y_s = [indexes_conn[i,1],indexes_poles[pole,1]]
        plt.plot(x_s,y_s,color='green',linewidth=1)
    
    #Save and Show
    ax.set_aspect('equal')
    plotname = "AllWiring_prob_%s_FullReliability.png" %(str(prob))
    plt.savefig(plotname, dpi=600)
    plt.show()
    
    
##=============================================================================
# Plot Pole Placement Solution
def PlotPoleSolutions(OnOff,indexes_poles,indexes_conn,indexes_excl,ConnPoles):
    cmap =  plt.cm.get_cmap('hsv', len(indexes_poles[:,0]))
        
    fig, ax = plt.subplots()
    for i in range(len(indexes_poles[:,0])):
        for j in range(len(ConnPoles[:,0])):
            if ConnPoles[j,0] == i:
                plt.scatter(indexes_conn[j, 0], indexes_conn[j, 1], s=1,c=cmap(i),marker='.')#, color=color)
        plt.scatter(indexes_poles[i,0],indexes_poles[i,1],s=3,c=cmap(i), marker= '^')
    ax.set_aspect('equal')
    plotname = "SolutionPlot_prob_%s_FullReliability.png" %(str(prob))
    plt.savefig(plotname, dpi=600)
    plt.show()

    fig, ax = plt.subplots()
    plt.scatter(indexes_excl[:,0],indexes_excl[:,1],s=1,c ='r',marker= 's')
    plt.scatter(indexes_poles[:,0],indexes_poles[:,1], s=1, c='b', marker='s')
    ax.set_aspect('equal')
    plotname = "ExclusionsPlotwSolnPoles_prob_%s_FullReliability.png" %(str(prob))
    plt.savefig(plotname, dpi=600)
    plt.show()
    
    PoleWiring(OnOff, indexes_poles)

    AllWiringPlot(OnOff, indexes_poles, indexes_conn, ConnPoles)          

##=============================================================================

#==============================================================================
# Calculate Closest Pole to POI (point of interconnection) to generation
def POI_Pole(lat_Generation,long_Generation,Long_exc_min,Lat_exc_min,d_EW_between,d_NS_between,indexes_poles):
    EW_dis = GPStoDistance(Lat_exc_min,Lat_exc_min,Long_exc_min,long_Generation)
    NS_dis = GPStoDistance(Lat_exc_min,lat_Generation,Long_exc_min,Long_exc_min)
    EW_index = int(EW_dis/d_EW_between)
    NS_index = int(NS_dis/d_NS_between)
    indexes_gen = [EW_index,NS_index]
    #Calculate distances between generation and pole
    #Individually feed in each pair
    num_poles = len(indexes_poles[:,0])
    Distance_Gen_Poles = np.zeros(num_poles)    
    for i in range(num_poles):
        Distance = DistanceBWindexes(indexes_gen,indexes_poles[i,:],d_EW_between,d_NS_between) #Type error: only size-1 arrays can be converted to Python Scalars
        Distance_Gen_Poles[i] = Distance
    closest_pole = np.argmin(Distance_Gen_Poles)
    
    return closest_pole
#==============================================================================

#==============================================================================
# Calc total load on poles
def PoleLoads(ConnPoles,num_poles):
    num_conns = len(ConnPoles[:,0])
    load_per_connection = kW_max/num_conns
    PoleLoad_matrix = np.zeros(num_poles)
    for i in range(num_poles):
        PoleLoad_matrix[i] = int(np.ceil(np.count_nonzero(ConnPoles[:,0]==i)))*load_per_connection        
    
    return PoleLoad_matrix
#==============================================================================
    
#==============================================================================
# Edges matrix: n-1 on line loss. |pole 1|pole 2|load loss|
def LineLosses(OnOff,ConnPoles,num_poles,Long_exc_min,Lat_exc_min,d_EW_between,d_NS_between,indexes_poles):
    #Load Lat and Long Gen
    lat_Generation = m.radians(Net_Parameters['lat_Generation'][0])
    long_Generation = m.radians(Net_Parameters['long_Generation'][0])
    POI = POI_Pole(lat_Generation,long_Generation,Long_exc_min,Lat_exc_min,d_EW_between,d_NS_between,indexes_poles)
    #Calculate load amount at each pole
    PoleLoad_matrix = PoleLoads(ConnPoles,num_poles)
    #Convert OnOff to list of edges (lines)
    num_edges = int(np.sum(OnOff)/2)
    Edges = np.zeros((num_edges,3))
    edge_count = 0
    for i in range(len(OnOff[:,0])):
        for j in range(i):
            if OnOff[i,j] == 1:
                OnOff_temp = np.copy(OnOff)
                OnOff_temp[i,j] = 0
                OnOff_temp[j,i] = 0
                #Calculate Load loss from this edge
                visited = np.zeros(num_poles)
                visited = Check_connections(POI,visited,OnOff_temp) #connected components to generation POI
                load_loss = 0
                for k in range(num_poles):
                    if visited[k] == 0:
                        load_loss = load_loss + PoleLoad_matrix[k]
                #Append Edges matrix with edge and load loss from loss of that edge (line)
                Edges[edge_count,:] = [i,j,load_loss]
                edge_count += 1
    return Edges
#==============================================================================   


#==============================================================================
# Internal connected components recursive. This works because undirected
def Check_connections(i,visited,OnOff):
    for j in range(len(OnOff[i,:])):
        if OnOff[i,j] == 1 and visited[j] == 0: #if j in a pole (999 is default for not a pole, and that pole is not visited)
            visited[j] = 1
            Check_connections(j,visited,OnOff)
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
# Wiring Algorithm
def WiringAlg(ConnPoles,prob, Cost_kWh, wiring_trans_cost, restoration_time,Long_exc_min,Lat_exc_min,indexes_poles,d_EW_between,d_NS_between,standalone):
    t0 = time.time()

    #Load solution pole indexes
    if standalone == "True":
        filename = "indexes_poles_reformatted_%s_prob_%s_soln.csv" %(str(reformatScaler),str(prob))
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
                #print("i")
                #print(i)
                #print("j")
                #print(j)
                #print("*********")
                ind = np.where(DistancesBWPoles == DistancesBWPoles_sorted[i,j]) #This is going to give two values, need adjust so only one half of matrix is considered
                #Place in use connection
                x = ind[0][0]
                y = ind[1][0]
                OnOff[x,y] = 1
                #make sure connection is also noted from the other pole's side (only will add up one side for total distances though so not double counting)
                OnOff[y,x] = 1
        # Check for islands
        components, visited = ConnectedComponents(OnOff) #need to make sure OnOff is being filled in way that there aren't rows of zeros 
        if components == 1:
            # Calculate Load loss risk
            Edges = LineLosses(OnOff,ConnPoles,num_poles,Long_exc_min,Lat_exc_min,d_EW_between,d_NS_between,indexes_poles)          
            total_load_loss_risk = sum(Edges[:,2])
            Reliability_Risk_Cost = total_load_loss_risk * prob * Cost_kWh * restoration_time
            if Reliability_Risk_Cost == 0:
                #Solve for total distance and calc cost
                total_distance = 0
                for i in range(num_poles):
                    for j in range(0,i):
                        if OnOff[i,j] == 1:
                            total_distance = total_distance + DistancesBWPoles[i,j]
                Wire_Cost = total_distance * wiring_trans_cost
                Best_Total_Cost = Wire_Cost+Reliability_Risk_Cost #save as current best
                goodToGo = 1
            else:
                num_conn_per_pole += 1
        else:
            num_conn_per_pole += 1 #This is occasionally exceeding number of poles, need to put in error fixing block
    Best_Reliability_Cost = np.copy(Reliability_Risk_Cost)
    Best_Wire_Cost = np.copy(Wire_Cost)
    Best_total_distance = np.copy(total_distance)   
    
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
        components, visited = ConnectedComponents(OnOff_temp)
        if components == 1: #no islands, Only move forward with best solution if no islands
            # Calculate Load loss risk
            Edges = LineLosses(OnOff_temp,ConnPoles,num_poles,Long_exc_min,Lat_exc_min,d_EW_between,d_NS_between,indexes_poles)
            total_load_loss_risk = sum(Edges[:,2])
            Reliability_Risk_Cost = total_load_loss_risk * prob * Cost_kWh * restoration_time
            if Reliability_Risk_Cost == 0:
                #Solve for total distance and calc cost
                total_distance = 0
                for i in range(num_poles):
                    for j in range(i):
                        if OnOff_temp[i,j] == 1:
                            total_distance = total_distance + DistancesBWPoles[i,j]
                Wire_Cost = total_distance * wiring_trans_cost
                Total_Cost = Wire_Cost+Reliability_Risk_Cost
                if Total_Cost < Best_Total_Cost: #Save best solution based on no islands and lowest cost
                    OnOff = np.copy(OnOff_temp)
                    Best_Total_Cost = np.copy(Total_Cost)
                    Best_Reliability_Cost = np.copy(Reliability_Risk_Cost)
                    Best_Wire_Cost = np.copy(Wire_Cost)
                    Best_total_distance = np.copy(total_distance) 
        #print(Best_Total_Cost)
        #print(Best_Reliability_Cost)
        #print("********************")
                
    t1 = time.time()
    total_time = t1-t0
    
    return Best_total_distance,Best_Wire_Cost,Best_Reliability_Cost,Best_Total_Cost, OnOff, DistancesBWPoles, num_conn_per_pole, total_time
    
#==============================================================================


if __name__ == "__main__":
    
    #Set Inputs for optimizations
    Net_Parameters = pd.read_excel('uGrid_Input.xlsx', sheet_name = 'Net')
    reformatScaler = int(Net_Parameters['reformatScaler'][0]) #parameter to decrease the resolution of image (speeds up processing)
    exclusionBuffer = int(Net_Parameters['exclusionBuffer'][0])#meters that poles need to be form exclusions (other poles, exclusions, and connections)
    MaxDistancePoleConn = int(Net_Parameters['MaxDistancePoleConn'][0])#(m) the maximum distance allowed for a pole to be from a connection
    minPoles = int(Net_Parameters['minPoles'][0])
    maxPoles = int(Net_Parameters['maxPoles'][0])
    range_limit = int(Net_Parameters['range_limit'][0])
    repeats = int(Net_Parameters['repeats'][0])
    global prob
    prob = Net_Parameters['prob'][0]
    Cost_kWh =  Net_Parameters['Cost_kWh'][0] #Change this to input from uGrid 
    restoration_time = int(Net_Parameters['restoration_time'][0])
    #Load all Econ input
    Econ_Parameters = pd.read_excel('uGrid_Input.xlsx', sheet_name = 'Econ')
    wiring_trans_cost = Econ_Parameters['Cost_Trans_wire'][0]/1000 #go from per km to per m
    
    LoadKW_MAK = pd.read_excel('LoadKW_MAK.xlsx',index_col=None, header=None)
    global kW_max
    kW_max = max(LoadKW_MAK[0])
    
    #Run Pole Placement Optimization, output is saved as csv files                                                                                                       
    Best_Reliability_Cost, OnOff, num_initial_clusters,ConnPole_soln, total_wire_distance_soln, total_trans_wire_distance_soln, indexes_poles_soln, totalCost_soln, indexes_conn, indexes_excl = PoleOpt(reformatScaler,minPoles,maxPoles,exclusionBuffer,range_limit,MaxDistancePoleConn,repeats,prob, Cost_kWh, wiring_trans_cost, restoration_time)    
    PlotPoleSolutions(OnOff,indexes_poles_soln,indexes_conn,indexes_excl,ConnPole_soln)
    
    if max(ConnPole_soln[:,1]) > MaxDistancePoleConn:
        print("Distances between houses and poles is too far, the farthest is "+str(max(ConnPole_soln[:,1]))+"m. Increase the maximum limit for number of poles")
    
    #Save total cost
    Costs = [Best_Reliability_Cost, totalCost_soln,total_wire_distance_soln, total_trans_wire_distance_soln]
    filename = "Costs_reformatted_%s_prob_%s_FullReliability_soln.csv" %(str(reformatScaler),str(prob))
    np.savetxt(filename,Costs, delimiter=",")
    
    #Run Wiring Optimization as Standalone with previous solution
    #total_distance, OnOff, DistancesBWPoles, num_conn_per_pole, total_time = WiringAlg(ConnPoles,Long_exc_min,Lat_exc_min,indexes_poles,d_EW_between,d_NS_between,"True")


    
