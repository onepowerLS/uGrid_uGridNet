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
# Create the exclusion arrary: captures the number of exclusions and the indexes of those exclusions
def ExclusionMapper(ExclusionMap_array,reformatScaler):
    y_index = []
    x_index = []
    i = 0
    while i < len(ExclusionMap_array[:,0]):
        #print(i)
        j = 0
        while j < len(ExclusionMap_array[0,:]):
            if ExclusionMap_array[i,j,0] != 255: #exclusion zones are both grey and black, safe zones are white (255,255,255)
                #Reformate Exclusion map to decreased resolution
                #Current resolution is to .15 m. Decreasing resolution to 1.5 m.
                #Current plan is to cut off bottom and right side of picture up to 1.5m (10 pixels)
                #if there are any exclusions in the new resolution square the whole square is marked as exclusion 
                new_y = int(i/reformatScaler)
                new_x = int(j/reformatScaler)
                #add saving index location
                y_index.append(new_y)
                x_index.append(new_x)
                if j < len(ExclusionMap_array[0,:]) - reformatScaler: #to not surpass index
                    i = i + reformatScaler -1 #jump ahead to next new index (-1 because of +1 at end of j while loop)
                else:
                    break
                if i < len(ExclusionMap_array[:,0]) - reformatScaler: #to not surpass index
                    j = j + reformatScaler #jump ahead to next new index
                else:
                    break
            else:
                j += 1
        i += 1
    indexes = np.transpose(np.array([x_index,y_index]))
    index_csv_name = "indexes_reformatted_%s.csv" %str(reformatScaler)
    np.savetxt(index_csv_name,indexes, delimiter=",")
    return  indexes
#==============================================================================


#=============================================================================
# Get Distance between array indexes
def DistanceBWindexes(indexesA,indexesB,d_EW_between,d_NS_between):
    A_sqr = np.zeros((len(indexesA), len(indexesB)))
    B_sqr = np.zeros((len(indexesA), len(indexesB)))
    for i in range(len(indexesA)):
        for j in range(len(indexesB)):
            A_sqr[i,j] = math.pow(((indexesA[i,0]-indexesB[j,0])*d_EW_between),2)
            #print(indexesA[i,0]-indexesB[j,0])
            #print((indexesA[i,0]-indexesB[j,0])*d_EW_between)
            #print(math.pow(((indexesA[i,0]-indexesB[j,0])*d_EW_between),2))
            B_sqr[i,j] = math.pow(((indexesA[i,1]-indexesB[j,1])*d_NS_between),2)
    DistanceAB = np.sqrt(A_sqr+B_sqr)
    return DistanceAB
#=============================================================================

#=============================================================================
# Randomly Place Poles with constraint of not being placed on other pole, connection, or exclusion
def RandomPolePlacement(indexes_conn,indexes_excl,num_poles,x_index_max,y_index_max,exclusion_buffer):
    #Maximum number of poles = number of connections
    #pole_max = len(indexes_conn[:,0]) #this is preset outside of function (in optimization)
    indexes_poles = np.zeros((num_poles,2))
    i = 0
    for i in range(num_poles):
        indexes_poles[i,:] = PolePlacementNoConflicts(indexes_conn,indexes_excl,indexes_poles,num_poles,x_index_max,y_index_max,exclusion_buffer)
    return indexes_poles
#==============================================================================
    
#==============================================================================
# Test new pole placement for conflicts
def PolePlacementNoConflicts(indexes_conn,indexes_excl,indexes_poles,num_poles,x_index_max,y_index_max,exclusion_buffer):
    #must have 20m buffer between exclusions, connections, and poles from new pole
    goodToGo = 0
    indexes_all_excl = np.concatenate((indexes_conn,indexes_excl,indexes_poles),axis=0)
    while goodToGo == 0:
        try_xy = np.array([[randint(0,x_index_max),randint(0,y_index_max)]])
        Distance_Pole_Excl_array = DistanceBWindexes(try_xy,indexes_all_excl,d_EW_between,d_NS_between)
        num_elements = len(Distance_Pole_Excl_array[:,0]*Distance_Pole_Excl_array[0,:])
        Distance_Pole_Excl_1xArray = np.reshape(Distance_Pole_Excl_array,(1,num_elements))
        for i in range(num_elements): #see if any of the distance are less than the exclusion buffer
            if Distance_Pole_Excl_1xArray[0,i] <= exclusion_buffer:
                goodToGo = 0 #Poles too close to exclusions, repeat
                break
            else:
                goodToGo = 1 #Poles far enough from exclusions
    return try_xy
#=============================================================================== 

#===============================================================================
# Match Connections to Poles Simplified Version
#Instead of trying to find a feasible solution, just create penalties that can be weighted in the optimization
def MatchConnectionsPolesSimple(indexes_conn,indexes_poles,d_EW_between,d_NS_between,MaxDistancePoleConn,num_poles):                   
    ConnPoles = np.zeros((len(indexes_conn[:,0]),2)) #first value is pole, second value is distance to that pole

    #Match Connections and Poles        
    DistanceConnPoles = DistanceBWindexes(indexes_conn,indexes_poles,d_EW_between,d_NS_between)
    for i in range(len(ConnPoles[:,0])):
        ConnPoles[i,0] = np.argmin(DistanceConnPoles[i,:])
        ConnPoles[i,1] = np.min(DistanceConnPoles[i,:])
    
    #Penalty Check 1: number of connections over allowable distance
    max_distance_penalty = 0
    for j in range(len(ConnPoles[:,0])):
        if ConnPoles[j,1] > MaxDistancePoleConn:
            max_distance_penalty += 1
    #Penalty Check 2: Exceeding allowable number of connections per pole
    max_connectionsPerPole_penalty = 0
    for j in range(num_poles):
        if np.count_nonzero(ConnPoles[:,0]==j) > PoleConnMax:
            max_connectionsPerPole_penalty += 1
    #Penalty Check 3: number of poles, want to reduce the number of poles
    poles_in_use = np.unique(ConnPoles[:,0])
    num_poles_in_use = len(poles_in_use)
    
        #Change pole 
        #if max_distance > MaxDistancePoleConn or max_PoleConn == 1:
            #randomly move one pole and complete restart
         #   changePole = randint(0,num_poles-1)
          #  indexes_poles[changePole,:] = PolePlacementNoConflicts(indexes_conn,indexes_excl,indexes_poles,num_poles,x_index_max,y_index_max,exclusion_buffer)
        #else:
         #   goodToGo == 1
        
    #Calculate the wire distances from all the poles and connections
    total_wire_distance = sum(ConnPoles[:,1])
    return ConnPoles, total_wire_distance, max_distance_penalty, max_connectionsPerPole_penalty,num_poles_in_use
#==============================================================================
             
        

#==============================================================================
# Match Connections to Poles
def MatchConnectionsPoles(indexes_conn,indexes_excl,indexes_poles,d_EW_between,d_NS_between,MaxDistancePoleConn,num_poles,x_index_max,y_index_max,exclusion_buffer,PoleConnMax):
    ConnPoles = np.zeros((len(indexes_conn[:,0]),2)) #first value is pole, second value is distance to that pole

    i = 0
    goodToGo = 0
    while goodToGo == 0:
        # First Calculate distances between connections and poles
        DistanceConnPoles = DistanceBWindexes(indexes_conn,indexes_poles,d_EW_between,d_NS_between)
        #Check 1) If any of the shortest distances are farther than the allowable distance from pole
        while i < len(ConnPoles[:,0]): #cycle through all connections
            ConnPoles[i,0] = np.argmin(DistanceConnPoles[i,:])
            ConnPoles[i,1] = np.min(DistanceConnPoles[i,:])
            if ConnPoles[i,1] > MaxDistancePoleConn:
                #randomly move one pole and complete restart
                changePole = randint(0,num_poles-1)
                indexes_poles[changePole,:] = PolePlacementNoConflicts(indexes_conn,indexes_excl,indexes_poles,num_poles,x_index_max,y_index_max,exclusion_buffer)
                i = 0
            else:
                i += 1
                #All connections to poles are under the allowable distance
        #Check 2) If there are no more than max allowable connections per pole
        j = 0
        while j < num_poles:
            if np.count_nonzero(ConnPoles[:,0]==j) > PoleConnMax: #If over limit get new connection assignment
                ConnPoles = ReassignConnection_PoleConnMaxConstraint(ConnPoles,j,DistanceConnPoles[j,:])
                if ConnPoles == 0: #This means no other second shortest distances where less than allowable distance from pole
                    #now need to randomly replace a random pole and try it all again
                    changePole = randint(0,num_poles-1)
                    indexes_poles[changePole,:] = PolePlacementNoConflicts(indexes_conn,indexes_excl,indexes_poles,num_poles,x_index_max,y_index_max,exclusion_buffer)
                    break #retry from the beginning (going back to Check 1)
                else: 
                    #successful reassignment of connection, recheck Check 2 from beginning
                    j = 0
            else:
                #No Poles have over the maximum allowable connections, moving forward
                j += 1
        #Check 3) If any of the poles are not being used
        k = 0
        while k < num_poles:
            if k in ConnPoles[:,0]: #Check if pole number is in the connections list
                k += 1 #pole has a connection, move on
            else:
                #Pole does not have a connection 
                #Check if closest connection to pole is under 50 m
                ConnPoles = ReassignConnection_PoleNoConnections(ConnPoles,j,DistanceConnPoles[:,k],MaxDistancePoleConn)
                if ConnPoles == 0:
                    #could not find connection for pole
                    #randomly reassign a pole and restart
                    changePole = randint(0,num_poles-1)
                    indexes_poles[changePole,:] = PolePlacementNoConflicts(indexes_conn,indexes_excl,indexes_poles,num_poles,x_index_max,y_index_max,exclusion_buffer)
                    break #retry from the beginning (going back to Check 1)
                
                else:
                    #successful reassignment, recheck Check 3
                    k = 0
        #If made it through all the checks (code runs to here) can calculate the total wire distances between poles and connections
        goodToGo = 1
    
    #Calculate the wire distances from all the poles and connections
    total_wire_distance = sum(ConnPoles[:,1])
    return ConnPoles, total_wire_distance
#==============================================================================        
                    
  
#==============================================================================
# Maximum Number of Connection per Pole Constraint hit, See if any of the connections can be reassigned
def ReassignConnection_PoleConnMaxConstraint(ConnPoles,j,ConnectionDistancesToPole,MaxDistancePoleConn):
    for i in range(len(ConnPoles[:,0])):
        if ConnPoles[i,1] == j:
            #Find second smallest distance to pole
            m1, m2 = float('inf'),float('inf')
            for x in ConnectionDistancesToPole:
                if x <= m1:
                    m1,m2 = x,m1
                elif x < m2:
                    m2 = x
            #Check if the second shortest is less than the allowable distance
            #if it is replace and return the new pole and distance
            if m2 < MaxDistancePoleConn:
                ConnPoles[i,1] = m2
                new_index = np.where(ConnectionDistancesToPole == m2)
                ConnPoles[i,0] = new_index[1][0]
                return ConnPoles
    #If none of the second distances are less than allowable return 0 to signal this error        
    return 0
#============================================================================== 

#==============================================================================
# Pole with no connections, reassign connection to empty pol
def ReassignConnection_PoleNoConnections(ConnPoles,j,PoleDistancesToConnections,MaxDistancePoleConn):
    min_poleConnDistance = np.min(PoleDistancesToConnections)
    if min_poleConnDistance < MaxDistancePoleConn:
        i = np.argmin(PoleDistancesToConnections)
        ConnPoles[i,1] = min_poleConnDistance
        ConnPoles[i,0] = j
        return ConnPoles #return revised solution
    return 0
#==============================================================================
     
                   
#=============================================================================
# Record to Record Travel Optimization
def RRT(PoleConnMax, deviation_factor,objpre,nrep,indexes_conn,indexes_poles,indexes_excl,d_EW_between,d_NS_between,MaxDistancePoleConn,num_poles,x_index_max,y_index_max,exclusion_buffer,np_penalty_factor,mc_penalty_factor,md_penalty_factor):
    #initialize RRT parameters
    record = np.copy(objpre)
    bestsoln =np.copy(indexes_poles)
    deviation = deviation_factor*record
    record_records = np.zeros(10000)
    
    for j in range(nrep):
        #Save old solution
        oldsoln = np.copy(indexes_poles)

        #Create new solution
        changePole = randint(0,num_poles-1) 
        indexes_poles[changePole,:] = PolePlacementNoConflicts(indexes_conn,indexes_excl,indexes_poles,num_poles,x_index_max,y_index_max,exclusion_buffer)
        
        #Evaluate new solution
        ConnPoles, total_wire_distance, max_distance_penalty, max_connectionsPerPole_penalty, num_poles_in_use = MatchConnectionsPolesSimple(indexes_conn,indexes_poles,d_EW_between,d_NS_between,MaxDistancePoleConn,num_poles)
        #Calculate Cost of Solution
        tempobj = PenaltiesToCost(total_wire_distance, num_poles_in_use, ConnPoles,PoleConnMax,num_poles)
        
        if tempobj < record+deviation:
            objnow = np.copy(tempobj)
            if objnow < record:
                record = np.copy(objnow)
                bestsoln = np.copy(indexes_poles)
                print("Best record is (total cost): " + str(record))
        else:
            indexes_poles = np.copy(oldsoln)
        
        record_records[j] = np.copy(record)
        
    return bestsoln, record, record_records 
#==============================================================================

#==============================================================================
# Calculate the Cost of the penalties to use as the minimizing optimization value
def PenaltiesToCost(total_wire_distance, num_poles_in_use, ConnPoles,PoleConnMax,num_poles):
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
    for j in range(num_poles):
        board_per_pole = int(np.count_nonzero(ConnPoles[:,0]==j)/PoleConnMax)
        num_dist_boards = num_dist_boards + board_per_pole
    total_distboard_cost = num_dist_boards*Cost_Dist_Board
    
    Total_cost = total_wire_cost + total_pole_cost + total_distboard_cost
    
    return Total_cost
#==============================================================================        
               

# Run Full Code ===============================================================
if __name__ == "__main__":
    
    ## Cycle through combinations of variable inputs to find best solution
    #First vary nrep and deviations
    devs = [0.005]
    nreps = [10000] #flattens out around 6000
    
    for deviation_factor in devs:
        for nrep in nreps:
            ## Variable Inputs    
            reformatScaler = 5 #parameter to decrease the resolution of image
            exclusion_buffer = 20 #meters that poles need to be form exclusions (other poles, exclusions, and connections)
            MaxDistancePoleConn = 50 #(m) the maximum distance allowed for a pole to be from a connection
            PoleConnMax = 20 #maximum number of connections allowed per pole
            #nrep = 500
            np_penalty_factor = 0
            mc_penalty_factor = 0
            md_penalty_factor = 0
            #deviation_factor = 0.05
        
            ## Initialization for Optimization
            t0 = time.time()
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
            #t1 = time.time()
            #total_time = t1-t0
            #print(total_time)
        
            #Random Pole Placement Initialization 
            #t0 = time.time()
            num_poles = len(indexes_conn[:,0])
            indexes_poles = RandomPolePlacement(indexes_conn,indexes_excl,num_poles,width,height,exclusion_buffer)
            #t1 = time.time()
            #total_time = t1-t0
            #print(total_time)
    
            #Test matching connections and poles simple function version
            #t0 = time.time()
            ConnPoles, total_wire_distance, max_distance_penalty, max_connectionsPerPole_penalty, num_poles_in_use = MatchConnectionsPolesSimple(indexes_conn,indexes_poles,d_EW_between,d_NS_between,MaxDistancePoleConn,num_poles)
            objpre = PenaltiesToCost(total_wire_distance, num_poles_in_use, ConnPoles,PoleConnMax,num_poles)
            #t1 = time.time()
            #total_time = t1-t0
            #print(total_time)
    
            ## Perform RRT
            #t0 = time.time()
            bestsoln_indexes_poles, record, record_records = RRT(PoleConnMax,deviation_factor,objpre,nrep,indexes_conn,indexes_poles,indexes_excl,d_EW_between,d_NS_between,MaxDistancePoleConn,num_poles,width,height,exclusion_buffer,np_penalty_factor,mc_penalty_factor,md_penalty_factor)
            #t1 = time.time()
            #total_time = t1-t0
            #print(total_time)    
            
            #Check RRT solution is accurate
            #t0 = time.time()
            ConnPoles, total_wire_distance, max_distance_penalty, max_connectionsPerPole_penalty, num_poles_in_use = MatchConnectionsPolesSimple(indexes_conn,bestsoln_indexes_poles,d_EW_between,d_NS_between,MaxDistancePoleConn,num_poles)
            record_check = PenaltiesToCost(total_wire_distance, num_poles_in_use, ConnPoles,PoleConnMax,num_poles)
            #Find max distance b/w connection and pole
            if max_distance_penalty > 0:
                max_dist = np.max(ConnPoles[:,1])
            #Find max number of connections per pole
            if max_connectionsPerPole_penalty > 0:
                max_conn_t = len(indexes_conn[:,0])
                for j in range(num_poles):
                    max_conn = np.count_nonzero(ConnPoles[:,0]==j)
                    if max_conn > max_conn_t:
                        max_conn_t = np.copy(max_conn)    
    
            #Print Outcomes
            if record_check == record:
                print("Solution found, total wire distance is: "+str(total_wire_distance))
                print("number of poles in use is: "+str(num_poles_in_use))
                if max_distance_penalty > 0:
                    print("Number of connections more than allowable distance from pole is: "+str(max_distance_penalty))
                    print("With maximum distance being: "+str(max_dist))
                if max_connectionsPerPole_penalty > 0:
                    print("Number of pole with more than the allowable number of connections is: "+str(max_connectionsPerPole_penalty))
                    print("With max number of connections per pole being: "+str(max_conn_t))
            else:
                print("There is a mismatch between the sanity check and RRT solution, debug please!")
            t1 = time.time()
            total_time = t1-t0
            print(total_time)
            
            ## Save combination of variable input results
            results = [record,max_distance_penalty,max_connectionsPerPole_penalty,num_poles_in_use,total_time]
            filename = "RRT_Results_wCost_WireSqr_nrep"+str(nrep)+"_devfactor"+str(deviation_factor)+".csv"
            np.savetxt(filename,results, delimiter=",")
            filename_records = "RRT_Record_Results_wCost_WireSqr_nrep"+str(nrep)+"_devfactor"+str(deviation_factor)+".csv"
            np.savetxt(filename_records,record_records, delimiter=",")
            filename_solution_indexes_poles = "indexes_poles_wCost_WireSqr_nrep"+str(nrep)+"_devfactor"+str(deviation_factor)+".csv"
            np.savetxt(filename_solution_indexes_poles,indexes_poles, delimiter=",")
            filename_ConnPoles = "ConnPoles_wCost_WireSqr_nrep"+str(nrep)+"_devfactor"+str(deviation_factor)+".csv" 
            np.savetxt(filename_ConnPoles,ConnPoles, delimiter=",")

    #Setup pole location optimization
    #Use Record to Record Travel Optimization
    #Pyomo does use external functions well in the objective function
    #Check uGrid Net rules 
    #constraints: (these are now set as penalties)
    # 1) can't be in exclusion zone
    # 2) minimum and maximum number of connections in proximity to pole
    # 3) minimum and maximum distnace of pole from connection
    #objective: main objective shortest overall wire, penalties can be added in for constraints and minimizing number of poles
    
    #Implement Record to Record Travel Optimization


    