# -*- coding: utf-8 -*-
"""
OpenSky network 2020 Dataset available from:
https://zenodo.org/record/3928550
The dataset utilized for this analysis pertains to the Jan 2020-April 2020 dataset found at the above link

@author: Rajat Bakshi

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import math 
from math import sin, cos, atan2, sqrt, radians
from datetime import datetime, timedelta
import geopandas as gpd
import folium
from folium import FeatureGroup
from geographiclib.geodesic import Geodesic
import ee
import geemap.eefolium as geemap
from folium.plugins import FloatImage


pd.options.mode.chained_assignment = None #disables 'copying' of dfs, rather modifies original df
pd.set_option('display.max_columns', None) #will display all columns instead of only displaying the first and last few

def getDistance(lat1,lon1,lat2,lon2): #gets distance in km given a pair of latitude and longitude coordinates
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c #distance

#uncomment if re-creating light dataframe from original data file
''' 
df=pd.read_csv("flightlist_20200401_20200430.csv")
print(df.head)
       

df = df.drop(df[df.origin ==0].index) #delete all rows that are missing origin airport ICAO code
df = df.drop(df[df.destination ==0].index) #delete all rows that are missing destination airport ICAO code


df.fillna(0,inplace=True)

liteDF=df.iloc[::80, :] # dataframe where only every 80th entry exists from original df (original csv is roughly 160mb in size)
liteDF.to_csv("liteData.csv")
'''
#comment if re-creating light dataframe from original data file
liteDF=pd.read_csv("liteData.csv") 
liteDF = liteDF.drop(liteDF.columns[[0]], axis=1)



altitudeDifference = liteDF.altitude_1-liteDF.altitude_2 #examine if altitude changes between altitude_1 and altitude_2 columns

plt.figure(1) #plots alt. difference from aircraft's first record to its last record over captured ATC data against randomized x-axis val.
plt.title('altitude difference data')
plt.scatter(np.random.randint(-50,50,size=(len(liteDF.altitude_1), 1)),altitudeDifference, alpha = 0.2)


flightDistance=[] #in km
flightTime=[] #in hours
flightDistance = [0 for i in range(len(liteDF))] 
flightTime = [0 for i in range(len(liteDF))] 
averageVelocity=[] #in km/h over duration of flight
averageVelocity = [0 for i in range(len(liteDF))] 
averageAltitude=[] #in ft over duratin of flight (Altitude_1 and Altitude_2 columns found to contain signficiant unreliable data)
averageAltitude = [0 for i in range(len(liteDF))] 
dayOne = datetime(1950, 1, 1, 6, 0, 0) #reference day
for i in range(len(liteDF)): #for each flight record in liteDF dataframe
    flightDistance[i]=getDistance(liteDF.iat[i,10],liteDF.iat[i,11],liteDF.iat[i,13],liteDF.iat[i,14]) #km
    firstSeen=datetime.strptime('+'.join(liteDF.iat[i,7].split('+', 1)[:1]),'%Y-%m-%d %H:%M:%S')
    lastSeen=datetime.strptime('+'.join(liteDF.iat[i,8].split('+', 1)[:1]),'%Y-%m-%d %H:%M:%S')
    flightTime[i]=((lastSeen-dayOne).total_seconds()-(firstSeen-dayOne).total_seconds())/3600 #in hours
    averageVelocity[i]=flightDistance[i]/flightTime[i] #in km/h
    averageAltitude[i]=(liteDF.iat[i,12]+liteDF.iat[i,15])/2*3.28084 #converting m to ft
    

plt.figure(2)
plt.title('average velocity data (km/h; ground speed) vs. altitude (ft)')
plt.scatter(averageVelocity,(liteDF.altitude_1+liteDF.altitude_2)/2*3.28084, alpha = 0.2)

plt.figure(3)
plt.xlim(0,20)
plt.title('average flight time (hours) vs. altitude (ft)')
plt.scatter(flightTime,(liteDF.altitude_1+liteDF.altitude_2)/2*3.28084, alpha = 0.2)

#Normalized values are used for k-means clustering
normalizedAltitude = [float(i)/max(averageAltitude) for i in averageAltitude]
normalizedVelocity = [float(i)/max(averageVelocity) for i in averageVelocity]
normalizedFlightDistance=[float(i)/max(flightDistance) for i in flightDistance]

#Adding newly calculated data columns to liteDF dataframe
liteDF['avg Altitude']=averageAltitude
liteDF['avg Velocity']=averageVelocity
liteDF['flight Distance']=flightDistance
liteDF['normalized Velocity']=normalizedVelocity
liteDF['normalized Altitude']=normalizedAltitude
liteDF['normalized Flight distance']=normalizedFlightDistance
liteDF.fillna(0,inplace=True)

#K-Means study for determing optimal number of clusters. To determine clusters from velocity and flight distance variable combinations
wss=[] #within-cluster sum of square errors
K=range(2,13) #testing cluster range

for k in K:
    km=KMeans(n_clusters=k)
    predictedKMclusters=km.fit_predict(liteDF[['normalized Velocity','normalized Flight distance']])
    wss.append(km.inertia_) #gets wss score

#Getting an "elbow method" plot to select optimal number of clusters
plt.figure(4)
plt.title("No. of clusters vs. WSS")
plt.plot(K,wss) 
plt.xlabel("No. of clusters (k)")
plt.ylabel("Within-Cluster Sum of Square Error (WSS) ")
plt.savefig("KMeansElbowMethodPlot.png",dpi=80) #To be loaded as floating image in map view
#Determined 5 clusters from elbow method plot

#Running KMeans clustering with 5 clusters
km=KMeans(n_clusters=5,random_state=0) #setting random state to an int makes it deterministic from run to run)
predictedKMclusters=km.fit_predict(liteDF[['normalized Velocity','normalized Flight distance']])
liteDF['cluster']=predictedKMclusters #storing cluster values into the working df

#Getting cluster-based dataframes
df1=liteDF[liteDF.cluster==0]
df2=liteDF[liteDF.cluster==1]
df3=liteDF[liteDF.cluster==2]
df4=liteDF[liteDF.cluster==3]
df5=liteDF[liteDF.cluster==4]


plt.figure(5)
plt.title('Average Flight Parameters (avg. true airspeed vs. flight distance)')
plt.xlabel('Average Speed (km/h)')
plt.ylabel('Flight Distance (km)')
plt.scatter(df1['avg Velocity'],df1['flight Distance'], color='magenta',label='Group 1',alpha = 0.2)
plt.scatter(df2['avg Velocity'],df2['flight Distance'], color='blue',label='Group 2',alpha = 0.2)
plt.scatter(df3['avg Velocity'],df3['flight Distance'], color='red',label='Group 3',alpha = 0.2)
plt.scatter(df4['avg Velocity'],df4['flight Distance'], color='darkorange',label='Group 4',alpha = 0.2)
plt.scatter(df5['avg Velocity'],df5['flight Distance'], color='blueviolet',label='Group 5',alpha = 0.2)
plt.legend()
plt.savefig("flightConditions.png",dpi=80) #To be loaded as floating image in map view
plt.show()


n=6 #6 most common values to show
print("cluster 1's common aircraft are",df1['typecode'].value_counts()[:n].index.tolist())
print("cluster 2's common aircraft are",df2['typecode'].value_counts()[:n].index.tolist())
print("cluster 3's common aircraft are",df3['typecode'].value_counts()[:n].index.tolist())
print("cluster 4's common aircraft are",df4['typecode'].value_counts()[:n].index.tolist())
print("cluster 5's common aircraft are",df5['typecode'].value_counts()[:n].index.tolist())

#interactiveWorldMap=folium.Map([20,0], zoom_start=3) #Use if not using geemap folium
interactiveWorldMap=geemap.Map(add_google_map=False) 
interactiveWorldMap.setCenter(0, 30, zoom=3)
FloatImage("flightConditions.png", bottom=20, left=75).add_to(interactiveWorldMap) #Floating image of cluster graph in front of map

geod = Geodesic.WGS84

#Creating layers for each cluster/group which can then be toggled in the map
feature_group1 = FeatureGroup(name='Group 1')
feature_group2 = FeatureGroup(name='Group 2')
feature_group3 = FeatureGroup(name='Group 3')
feature_group4 = FeatureGroup(name='Group 4')
feature_group5 = FeatureGroup(name='Group 5')

def getFlightPath_AntiMeridianIntersectIndex(points): #Detects whether flight patch (stored in points) crosses the anti-meridian (180 deg long. line)
    #Returns the index of the last point that hasn't crossed the anti-meridian line. 
    #If path doesn't cross, -1 returned.
    
    for i in range(len(points)-1):
        if abs(points[i][1]-points[i+1][1])>300:
            return i
        
    return -1

def getFlightPathOverPolesPointIndex(points): #Detects whether any points in flight path are in northern/southern extremes (+/- 75 deg. lat.)
    #Returns the index of the first point detected in the extremities.
    #Returns -1 if flight path doesn't visit the latitudinal extremities.
    
    for i in range(len(points)-1):
        if abs(points[i][0])>75:
            #if firstNorthernPointFound==False:
                #firstNorthernPointIndex=i
            #newPoints.pop(i);
            if abs(points[i][1]-points[i+1][1])>280:
                return i
    
    return -1

def plotGreatCircleFlightPath(lat1,lon1,lat2,lon2,cluster): #Plots "great circle" flight path between two coordinates given long. and lat.
    #Flight path stored in pathPoints array of 'num' discrete points (point-to-point straight lines approximating the curved path on flat map)
    if cluster == "cluster1": 
        pathColor="magenta" #Sets path color for the cluster that the path belongs to per the legend defined in the Avg. Flight Parameters cluster plot
        childLayer=feature_group1 #polylines of cluster1 flight paths will be stored in feature_group1 layer of the map.
    elif cluster == "cluster2":
        pathColor="blue"
        childLayer=feature_group2
    elif cluster == "cluster3":
        pathColor="red" 
        childLayer=feature_group3
    elif cluster == "cluster4":
        pathColor="darkorange"  
        childLayer=feature_group4
    else:
        pathColor="blueviolet" 
        childLayer=feature_group5
    
    g = geod.Inverse(lat1,lon1,lat2,lon2)
    l = geod.Line(g['lat1'], g['lon1'], g['azi1'])
    num = 12 #Discretizing curved path into 12 point-to-point straight lines
    pathPoints=[] #Stores curved path points
    for i in range(num+1): #Calculating and storing path points
        pos = l.Position(i * g['s12'] / num)
        pathPoints.append((pos['lat2'], pos['lon2']))
        
    flightPath_AntiMeridianTntersectIndex=getFlightPath_AntiMeridianIntersectIndex(pathPoints) #Storing point index before flight path crosses
    #anti-meridian line. If flight path never crosses, -1 is stored.

    flightPathOverPolesPointIndex=getFlightPathOverPolesPointIndex(pathPoints) #Storing the index of first point detected in the high lat. regions (+/-75 deg.)
    #If no point found, -1 is stored.
    
    if flightPath_AntiMeridianTntersectIndex!=-1 and flightPathOverPolesPointIndex==-1: #If flight path crosses anti-meridian and doesn't go in high lat. regions,
        #The flight path will be drawn on the map with two polylines, one before crossing, and one after crossing. Without breaking the polylines, long
        #horizontal streaks from -180 anti-meridian to 180 anti-meridian would've been rendered due to the continuity of PolyLine function.
        folium.PolyLine(pathPoints[0:(flightPath_AntiMeridianTntersectIndex)], color=pathColor, weight=1, opacity=0.5).add_to(childLayer)
        folium.PolyLine(pathPoints[(flightPath_AntiMeridianTntersectIndex+1):(len(pathPoints))], color=pathColor, weight=1, opacity=0.5).add_to(childLayer)
    
    elif flightPathOverPolesPointIndex!=-1: #If flight path doesn't cross anti-meridian but is in the high lat. regions, path will be comprised of
        #two polylines, one from origin to the first point of flightpath that is in the high lat. region. The second polyline being a mirror of it.
        #Still trying to get this to work right.
        
        folium.PolyLine(pathPoints[0:(flightPathOverPolesPointIndex-2)], color=pathColor, weight=1, opacity=0.5).add_to(childLayer) #first polyline
        #folium.PolyLine(pathPoints[(len(pathPoints)-flightPathOverPolesPointIndex+2):len(pathPoints)], color="black", weight=2.5, opacity=0.3).add_to(childLayer)
        #Second polyline^^; renders a horizontal streak if enabled.
        
    else: #Flight path neither crosses anti-meridian, or enters high lat. region. 
        folium.PolyLine(pathPoints, color=pathColor, weight=1, opacity=0.5).add_to(childLayer)
    
    return

#Plots "great circle" flight path from origin to destination lat. long. coordinates (shortest path from the origin to destination coordinate on globe)
#On flat map projections, such a flight path results looks curved.
for i in range(len(df1)):
    plotGreatCircleFlightPath(df1.iat[i,10],df1.iat[i,11],df1.iat[i,13],df1.iat[i,14],"cluster1")
    
for i in range(len(df2)):
    plotGreatCircleFlightPath(df2.iat[i,10],df2.iat[i,11],df2.iat[i,13],df2.iat[i,14],"cluster2")

for i in range(len(df3)):
    plotGreatCircleFlightPath(df3.iat[i,10],df3.iat[i,11],df3.iat[i,13],df3.iat[i,14],"cluster3")
    
for i in range(len(df4)):
    plotGreatCircleFlightPath(df4.iat[i,10],df4.iat[i,11],df4.iat[i,13],df4.iat[i,14],"cluster4")
    
for i in range(len(df5)):
    plotGreatCircleFlightPath(df5.iat[i,10],df5.iat[i,11],df5.iat[i,13],df5.iat[i,14],"cluster5")

#Adding layers to main map
interactiveWorldMap.add_child(feature_group1)
interactiveWorldMap.add_child(feature_group2)
interactiveWorldMap.add_child(feature_group3)
interactiveWorldMap.add_child(feature_group4)
interactiveWorldMap.add_child(feature_group5)

#layer control
interactiveWorldMap.add_child(folium.map.LayerControl())    

#Storing common aircraft list for each cluster in respective list
cluster1Aircraft=df1['typecode'].value_counts()[:n].index.tolist()
cluster2Aircraft=df2['typecode'].value_counts()[:n].index.tolist()
cluster3Aircraft=df3['typecode'].value_counts()[:n].index.tolist()
cluster4Aircraft=df4['typecode'].value_counts()[:n].index.tolist()
cluster5Aircraft=df5['typecode'].value_counts()[:n].index.tolist()

#Removing the most popular "aircraft" ("0"; representative of NaN/blank values) from the list of common aircraft for each cluster
cluster1Aircraft.pop(0)
cluster2Aircraft.pop(0)
cluster3Aircraft.pop(0)
cluster4Aircraft.pop(0)
cluster5Aircraft.pop(0)

labels=[] #Labels/textual data for legend. To list 5 most common aircraft for each cluster
labels.append(("Group 1:"+" "+cluster1Aircraft[0]+", "+cluster1Aircraft[1]+", "+cluster1Aircraft[2]+", "+cluster1Aircraft[3]+", "+cluster1Aircraft[4]))
labels.append(("Group 2:"+" "+cluster2Aircraft[0]+", "+cluster2Aircraft[1]+", "+cluster2Aircraft[2]+", "+cluster2Aircraft[3]+", "+cluster2Aircraft[4]))
labels.append(("Group 3:"+" "+cluster3Aircraft[0]+", "+cluster3Aircraft[1]+", "+cluster3Aircraft[2]+", "+cluster3Aircraft[3]+", "+cluster3Aircraft[4]))
labels.append(("Group 4:"+" "+cluster4Aircraft[0]+", "+cluster4Aircraft[1]+", "+cluster4Aircraft[2]+", "+cluster4Aircraft[3]+", "+cluster4Aircraft[4]))
labels.append(("Group 5:"+" "+cluster5Aircraft[0]+", "+cluster5Aircraft[1]+", "+cluster5Aircraft[2]+", "+cluster5Aircraft[3]+", "+cluster5Aircraft[4]))

colors=[(255, 51, 236),(51, 73, 255),(255, 51, 51),(255, 116, 51),(131, 51, 255)] #Color field for each cluster in the legend. Colors match cluster colors previously defined.
interactiveWorldMap.add_legend(title="Common aircraft per group (By Typecode. Ex: B748=B747-8):",labels=labels, position ='bottomleft',colors=colors)

interactiveWorldMap.save("index.html") #Contains interactive map


















