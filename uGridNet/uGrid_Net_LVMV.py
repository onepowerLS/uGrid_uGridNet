# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:13:33 2019

@author: Phy
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 09:20:26 2019

uGrid Net
The goal is to take the kml file which has highlighted polygons of "can't build here"
areas and also an inputted generation station gps location, and determine where
to place distribution poles and how to connect the distribution network together.

This creates a LV 220V and MV 6.3kV network layout, where the MV works as a backbone.

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
def Clustering(indexes_conn,num_clusters):
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
def PolePlacement(reformatScaler,num_clusters,exclusionBuffer,range_limit,indexes_conn,indexes_excl,height,width,d_EW_between,d_NS_between):                    
      
    #Get initial solution of pole indexes with Gaussian Mean Clustering
    #The mean of the clusters is the initial pole placement
    ConnPole_initial, initial_pole_placement = Clustering(indexes_conn,num_clusters)
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
       
    return ConnPoles, indexes_poles_in_use_og

#==============================================================================
             
 
#==============================================================================
# Calculate the Cost of the penalties to use as the minimizing optimization value
def PenaltiesToCost(conn_wiring, LV_wiring, MV_wiring, num_LV_poles, num_MV_poles, ConnPoles):
    #Load all Econ input
    Econ_Parameters = pd.read_excel('uGrid_Input.xlsx', sheet_name = 'Econ')

    #Pull out costs needed for penalties
    Cost_Conn_Wire = Econ_Parameters['Cost_Housing_Wiring'][0]/1000 #converting per km to per m 
    Cost_LV_Wire = Econ_Parameters['Cost_Dist_wire'][0]/1000 #converting per km to per m 
    Cost_MV_Wire = Econ_Parameters['Cost_Trans_wire'][0]/1000 #converting per km to per m 
    Cost_Pole = Econ_Parameters['Cost_Pole'][0]
    Cost_StepDown_Trans = Econ_Parameters['Cost_Pole_Trans'][0]
    Cost_StepUp_Trans = Econ_Parameters['Cost_Step_up_Trans'][0]

    #Pole Cost
    Total_Cost_Pole = (num_LV_poles+num_MV_poles+1)*Cost_Pole #Add plus one pole for POI of gen pole
    
    #Wiring Costs
    Total_Cost_Conn_Wiring = conn_wiring*Cost_Conn_Wire
    Total_Cost_LV_Wiring = LV_wiring*Cost_LV_Wire
    Total_Cost_MV_Wiring = MV_wiring*Cost_MV_Wire
    Total_All_Wiring = Total_Cost_Conn_Wiring+Total_Cost_LV_Wiring+Total_Cost_MV_Wiring
    
    #Transformer Costs
    #Step-down (one at each MV pole, excluding POI of generation)
    Total_Cost_StepDown_Trans = num_MV_poles*Cost_StepDown_Trans
    Total_Trans_Cost = Total_Cost_StepDown_Trans+Cost_StepUp_Trans #one step-up trans at POI of gen.     
    
    #Meter Costs
    meter_cost_poles= []
    for j in range(num_LV_poles):
        conn_per_pole = int(np.ceil(np.count_nonzero(ConnPoles[:,0]==j)))
        meter_cost = -0.0042*conn_per_pole**5 + 0.1604*conn_per_pole**4 - 2.3536*conn_per_pole**3 + 16.776*conn_per_pole**2 - 59.5*conn_per_pole + 111.25
        meter_cost_poles.append(meter_cost)
    Total_meter_cost = sum(meter_cost_poles)
    
    Total_cost = Total_meter_cost + Total_Trans_Cost + Total_All_Wiring + Total_Cost_Pole
    
    return Total_cost
#==============================================================================        
 
    
#==============================================================================
# Collect the connection and exclusions data
def CollectVillageData():
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
            
    return indexes_conn, indexes_excl, height, width, d_EW_between, d_NS_between,Long_exc_max, Long_exc_min,Lat_exc_max, Lat_exc_min

#==============================================================================
 
#==============================================================================
# Plot All Poles and All wiring
def Plot_AllPoles_AllWiring(POI,OnOff_MV, indexes_poles_MV, OnOff_groups, indexes_poles_groups, indexes_conn, ConnPoles, indexes_poles_LV_all):
    
    #MV Poles and Wiring
    num_poles_MV = len(indexes_poles_MV[:,0])
    goodToGo = 0
    for i in range(num_poles_MV):
        for j in range(i):
            if OnOff_MV[i,j] == 1:
                match = np.array([[indexes_poles_MV[i,0],indexes_poles_MV[i,1]],[indexes_poles_MV[j,0],indexes_poles_MV[j,1]]]) 
                if goodToGo == 0:
                    goodToGo = 1
                    wiringMatrix_MV = np.copy(match)
                else:
                    wiringMatrix_MV = np.concatenate((wiringMatrix_MV,match),axis=1) 
    #MV Plot
    num_match_MV = len(wiringMatrix_MV[0,:])
    fig, ax = plt.subplots()
    for i in range(0,num_match_MV-1,2):
        j = i + 1
        plt.plot(wiringMatrix_MV[:,i],wiringMatrix_MV[:,j],color='green',linewidth=2)
    plt.scatter(indexes_poles_MV[:,0],indexes_poles_MV[:,1],s=2,c='b')
    #Plot POI with big black dot
    plt.scatter(POI[0],POI[1],s=5,c='black', marker='s')
    
    #LV Poles and Wiring
    cmap =  plt.cm.get_cmap('hsv', len(indexes_poles_groups))
    for group in range(len(indexes_poles_groups)):
        indexes_group = indexes_poles_groups[group]
        OnOff_group = OnOff_groups[group]
        num_poles_group = len(indexes_group)
        if len(indexes_group) != 1: #if only one index, skip (it will be under MV pole anyway)
            goodToGo = 0
            for i in range(num_poles_group):
                for j in range(i):
                    if OnOff_group[i,j] == 1:
                        match = np.array([[indexes_group[i][0],indexes_group[i][1]],[indexes_group[j][0],indexes_group[j][1]]]) 
                        if goodToGo == 0:
                            goodToGo = 1
                            wiringMatrix_group = np.copy(match)
                        else:
                            wiringMatrix_group = np.concatenate((wiringMatrix_group,match),axis=1) 
            #LV Plot
            num_match_group = len(wiringMatrix_group[0,:])
            for i in range(0,num_match_group-1,2):
                j = i + 1
                plt.plot(wiringMatrix_group[:,i],wiringMatrix_group[:,j],color = cmap(group),linewidth=1.5)
            for k in range(num_poles_group):
                plt.scatter(indexes_group[k][0],indexes_group[k][1],s=2,c='g')
            
        #Connections wiring
        num_conn = len(indexes_conn[:,0])
        for i in range(num_conn):
            pole = int(ConnPoles[i,0])
            x_s = [indexes_conn[i,0],indexes_poles_LV_all[pole,0]]
            y_s = [indexes_conn[i,1],indexes_poles_LV_all[pole,1]]
            plt.plot(x_s,y_s,color='orange',linewidth=1)
        
    ax.set_aspect('equal')
    plotname = "AllPolesAllWiring.png"
    plt.savefig(plotname)
    plt.show()
#==============================================================================
        
    
#==============================================================================
# Internal connected components recursive. This works because undirected
def Check_connections(i,visited,OnOff):
    for j in range(len(OnOff[i,:])):
        if OnOff[i,j] == 1 and visited[j] == 0: #if j in a pole (999 is default for not a pole, and that pole is not visited)
            visited[j] = 1
            Check_connections(j,visited,OnOff)
        #else continue to next pole
    #Check OnOff from other direction
    #for k in range(len(OnOff[:,i])):
    #    if OnOff[k,i] == 1 and visited[k] == 0: #if j in a pole (999 is default for not a pole, and that pole is not visited)
    #        visited[k] = 1
    #        Check_connections(k,visited,OnOff)
    
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
def WiringAlg(indexes_poles,d_EW_between,d_NS_between,standalone):
    t0 = time.time()

    #Load solution pole indexes
    if standalone == "True":
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
    num_conn_per_pole = 1 #starting value
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
        components, visited = ConnectedComponents(OnOff_temp)
        #PoleWiring(OnOff_temp, indexes_poles) #check solution
        if components == 1: #no islands and keeping meshed network, change OnOff to remove that connection
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
    
    return total_distance, OnOff, DistancesBWPoles, num_conn_per_pole, total_time
    
#==============================================================================

#==============================================================================
# Load behind each MV pole, to check that not over 220V line limit
def loadBehindPoles(ConnPoles_MV, indexes_poles_HV,ConnPoles_LV, indexes_poles_LV,num_conns,MV_pole_num,load_per_conn):
    ConnPoles_LVMV = np.zeros(num_conns) #the MV pole that a house connects to
    load_behind_MV_poles = np.zeros(MV_pole_num)
    for i in range(num_conns):
        Conn = int(ConnPoles_LV[i,0])
        ConnPoles_LVMV[i] = ConnPoles_MV[Conn,0]
    for j in range(MV_pole_num):
        load_behind_MV_poles[j] = np.count_nonzero(ConnPoles_LVMV==j)*load_per_conn
    max_load_behind = max(load_behind_MV_poles)
    return max_load_behind, ConnPoles_LVMV
#==============================================================================        
    
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
    
    return closest_pole, indexes_gen
#==============================================================================


if __name__ == "__main__":
    
    t0 = time.time()
    
    #Set Inputs for optimizations
    Net_Parameters = pd.read_excel('uGrid_Input.xlsx', sheet_name = 'Net')
    reformatScaler = int(Net_Parameters['reformatScaler'][0]) #parameter to decrease the resolution of image (speeds up processing)
    exclusionBuffer = int(Net_Parameters['exclusionBuffer'][0])#meters that poles need to be form exclusions (other poles, exclusions, and connections)
    MaxDistancePoleConn = int(Net_Parameters['MaxDistancePoleConn'][0])#(m) the maximum distance allowed for a pole to be from a connection
    minPoles = 60#int(Net_Parameters['minPoles'][0])
    maxPoles = 90#int(Net_Parameters['maxPoles'][0])
    range_limit = int(Net_Parameters['range_limit'][0])
    repeats = int(Net_Parameters['repeats'][0])
    
    #Load Lat and Long Gen
    lat_Generation = m.radians(Net_Parameters['lat_Generation'][0])
    long_Generation = m.radians(Net_Parameters['long_Generation'][0])
    
    repeats_MV_clusters = 5
    repeats_LV_clusters = 5
    
    repeats_LV_poles = 5
    repeats_MV_poles = 5
    
    Best_Total_Cost = 99999999999999999999
    
    #Community Load Data
    LoadKW_MAK = pd.read_excel('LoadKW_MAK.xlsx',index_col=None, header=None)
    global kW_max
    kW_max = max(LoadKW_MAK[0])
    
    #Line Data
    LV = 220 #V
    MV = 6300 #V
    LV_kW = (220*130)/1000 #kW
    MV_kW = 1.2*1000 #kW
    LV_safetyfactor = 0.5
    LV_kW_safety = LV_kW*LV_safetyfactor
    
    #Collect Village Data (connections and exclusion zones)
    indexes_conn, indexes_excl, height, width, d_EW_between, d_NS_between,Long_exc_max, Long_exc_min,Lat_exc_max, Lat_exc_min = CollectVillageData()
    num_conns = len(indexes_conn[:,0])
    load_per_conn = kW_max/num_conns
    
    Best_Total_Cost_List = []
    
    #Cycle through for best solution
    for i in range(minPoles,maxPoles):
        for nrep_LV in range(repeats_LV_poles):
            #Clustering for LV Poles
            ConnPoles_LV, indexes_poles_LV = PolePlacement(reformatScaler,i,exclusionBuffer,range_limit,indexes_conn,indexes_excl,height,width,d_EW_between,d_NS_between)                    
            LV_pole_num = len(indexes_poles_LV[:,0])
            #Calculate the minimum number of MV Poles
            min_MV_poles = int(kW_max/LV_kW)
        
            for nrep_MV in range(repeats_MV_poles):
                print("Number of Poles: "+str(i)+",and LV repeat is "+str(nrep_LV+1)+" out of "+str(repeats_LV_poles)+", MV repeat is "+str(nrep_MV+1)+" out of "+str(repeats_MV_poles)+".") 
                t_temp = time.time()
                #Clustering for MW Poles
                load_behind = 99999999999999
                MV_pole_num = np.copy(min_MV_poles)
                while load_behind > LV_kW_safety:
                    MV_pole_num += 1 # add MV pole until load behind is below limits
                    for repeat_num in range(repeats_MV_clusters):
                        ConnPoles_MV, indexes_poles_MV = PolePlacement(reformatScaler,MV_pole_num,exclusionBuffer,range_limit,indexes_poles_LV,indexes_excl,height,width,d_EW_between,d_NS_between)  
                        #Calculate load behind by clusters
                        load_behind, ConnPoles_LVMV = loadBehindPoles(ConnPoles_MV, indexes_poles_MV,ConnPoles_LV, indexes_poles_LV,num_conns,MV_pole_num,load_per_conn)
                        if load_behind < LV_kW_safety:
                            break #Load behind the poles is below the limit, continue to wiring layout
                
                ##Plot poles to make sure it makes sense
                #cmap =  plt.cm.get_cmap('hsv', MV_pole_num)
                #fig, ax = plt.subplots()
                #for i in range(MV_pole_num):
                #    for j in range(LV_pole_num):
                #        if ConnPoles_MV[j,0] == i:
                #            plt.scatter(indexes_poles_LV[j, 0], indexes_poles_LV[j, 1], s=1,c=cmap(i),marker='.')#, color=color)
                #    plt.scatter(indexes_poles_MV[i,0],indexes_poles_MV[i,1],s=3,c=cmap(i), marker= '^')
                #ax.set_aspect('equal')
                #plt.show()

                #Calculate HV Wiring Layout
                #Find POI of generation
                closest_pole, indexes_gen = POI_Pole(lat_Generation,long_Generation,Long_exc_min,Lat_exc_min,d_EW_between,d_NS_between,indexes_poles_MV)
                #add POI_MV to wiring for MV
                indexes_Poles_MV_wPOI = np.concatenate((indexes_poles_MV, [indexes_gen]), axis=0)
                total_distance_MV, OnOff_MV, DistancesBWPoles_MV, num_conn_per_pole_MV, total_time_MV = WiringAlg(indexes_Poles_MV_wPOI,d_EW_between,d_NS_between,"False")
        
                #Calculate LV Wiring Layout
                #Group indexes according to connecting MV pole
                group_indexes = []
                group_poles = []
                for MV_pole in range(MV_pole_num):
                    temp_group_indexes = []
                    temp_group_poles = []
                    for LV_pole in range(LV_pole_num):
                        if ConnPoles_MV[LV_pole,0] == MV_pole:
                            temp_group_indexes.append(indexes_poles_LV[LV_pole,:])
                            temp_group_poles.append(LV_pole)
                    temp_group_indexes.append(indexes_poles_MV[MV_pole,:]) #add MV pole location to end
                    group_indexes.append(temp_group_indexes)
                    group_poles.append(temp_group_poles)
                #Determine wiring layout
                OnOff_groups = []
                DistanceBWPoles_groups = []
                total_distance_LV = 0
                for MV_pole_group in range(MV_pole_num):
                    total_distance_temp, OnOff_temp, DistancesBWPoles_temp, num_conn_per_pole_temp, total_time_temp = WiringAlg(np.array(group_indexes[MV_pole_group]),d_EW_between,d_NS_between,"False")
                    #Add group OnOff to total OnOff
                    OnOff_groups.append(OnOff_temp)
                    DistanceBWPoles_groups.append(DistancesBWPoles_temp)
                    total_distance_LV = total_distance_LV + total_distance_temp
                    
                #Calculate Costs
                #Connection Total Distance
                conn_wiring = sum(ConnPoles_LV[:,1])
                Total_Cost = PenaltiesToCost(conn_wiring, total_distance_LV, total_distance_MV, LV_pole_num, MV_pole_num, ConnPoles_LV)
                #Save Best Solution Based on Cost
                if Total_Cost < Best_Total_Cost and max(ConnPoles_LV[:,1]) < MaxDistancePoleConn:
                    #Save Best Solutions
                    Best_OnOff_MV = np.copy(OnOff_MV)
                    Best_indexes_poles_MV = np.copy(indexes_Poles_MV_wPOI)
                    Best_OnOff_groups = np.copy(OnOff_groups)
                    Best_group_indexes = np.copy(group_indexes)
                    Best_ConnPoles_LV = np.copy(ConnPoles_LV)
                    Best_indexes_poles_LV = np.copy(indexes_poles_LV)
                    Best_ConnPoles_MV = np.copy(ConnPoles_MV)
                    Best_Total_Cost = np.copy(Total_Cost)
                    Best_Total_Cost_List.append(Total_Cost)
                print("This iteration's total cost is "+str(Total_Cost)+" and the best total cost is "+str(Best_Total_Cost)+".")
                print("This iteration's maximum distance between a house and a LV pole is "+str(max(ConnPoles_LV[:,1]))+" and the allowable distance is "+str(MaxDistancePoleConn)+".")
                temp_time = time.time() - t_temp
                print("This iteration's calculation time is "+str(temp_time)+"s.")
                print("************************************************************************")
        
    #Plot layouts
    Plot_AllPoles_AllWiring(indexes_gen, Best_OnOff_MV, Best_indexes_poles_MV, Best_OnOff_groups, Best_group_indexes, indexes_conn, Best_ConnPoles_LV, Best_indexes_poles_LV)
    
    #Save all solutions to csv's
    filename = "Best_OnOff_MV_%s_soln.csv" %(str(reformatScaler))
    np.savetxt(filename,Best_OnOff_MV, delimiter=",")
    filename = "Best_indexes_poles_MV_%s_soln.csv" %(str(reformatScaler))
    np.savetxt(filename,Best_indexes_poles_MV, delimiter=",")
    for group_save in range(len(Best_group_indexes)):
        filename = "Best_OnOff_groups_%s_soln_%s.csv" %(str(reformatScaler),str(group_save))
        np.savetxt(filename,Best_OnOff_groups[group_save], delimiter=",")
        filename = "Best_group_indexes_%s_soln_%s.csv" %(str(reformatScaler),str(group_save))
        np.savetxt(filename,Best_group_indexes[group_save], delimiter=",")
    filename = "Best_ConnPoles_LV_%s_soln.csv" %(str(reformatScaler))
    np.savetxt(filename,Best_ConnPoles_LV, delimiter=",")
    filename = "Best_indexes_poles_LV_%s_soln.csv" %(str(reformatScaler))
    np.savetxt(filename,Best_indexes_poles_LV, delimiter=",")
    filename = "Best_ConnPoles_MV_%s_soln.csv" %(str(reformatScaler))
    np.savetxt(filename,Best_ConnPoles_MV, delimiter=",")
    filename = "Best_Total_Cost_%s_soln.csv" %(str(reformatScaler))
    np.savetxt(filename,Best_Total_Cost_List, delimiter=",")
    
    t1 = time.time()
    total_time = t1-t0
    print("The total calculation time is "+str(total_time)+".")

            
            
                    
    


    
