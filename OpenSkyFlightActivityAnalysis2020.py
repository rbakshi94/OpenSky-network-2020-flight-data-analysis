# -*- coding: utf-8 -*-
"""
OpenSky network 2020 Dataset available from:
https://zenodo.org/record/3928550
@author: Rajat Bakshi
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import math 
from math import sin, cos, atan2, sqrt, radians
from datetime import datetime, timedelta

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

    

df=pd.read_csv("flightlist_20200401_20200430.csv")
print(df.head)

plt.figure(0) #plots flight path in lat and long.
plt.scatter(df.latitude_1,df.longitude_1)

"""
plt.figure(0) #plots flight path in lat and long.
plt.scatter(np.random.randint(-50,50,size=(len(df.altitude_1), 1)),df.altitude_1, alpha = 0.2)

plt.figure(0) #plots flight path in lat and long.
plt.scatter(np.random.randint(-50,50,size=(len(df.altitude_1), 1)),df.altitude_2, alpha = 0.2)
"""


df.fillna(0,inplace=True)
liteDF=df.iloc[::500, :] # dataframe where only every 500th entry exists from original df


plt.figure(0) #plots altitude_1 data against random x-axis values
plt.title('altitude_1 data')
plt.scatter(np.random.randint(-50,50,size=(len(liteDF.altitude_1), 1)),liteDF.altitude_1, alpha = 0.2)


plt.figure(0) #plots altitude_2 data against random x-axis values
plt.title('altitude_2 data')
plt.scatter(np.random.randint(-50,50,size=(len(liteDF.altitude_1), 1)),liteDF.altitude_2, alpha = 0.2)

altitudeDifference = liteDF.altitude_1-liteDF.altitude_2


plt.figure(1) #plots alt. difference from aircraft's first record to its last record over captured ATC data
plt.title('altitude difference data')
plt.scatter(np.random.randint(-50,50,size=(len(liteDF.altitude_1), 1)),altitudeDifference, alpha = 0.2)


distanceTravelled=[] #in km
travelledTime=[] #in hours
distanceTravelled = [0 for i in range(len(liteDF))] 
travelledTime = [0 for i in range(len(liteDF))] 
averageVelocity=[] #in km/h
averageVelocity = [0 for i in range(len(liteDF))] 
averageAltitude=[]
averageAltitude = [0 for i in range(len(liteDF))] 
dayOne = datetime(1950, 1, 1, 6, 0, 0)
for i in range(len(liteDF)):
    distanceTravelled[i]=getDistance(liteDF.iat[i,10],liteDF.iat[i,11],liteDF.iat[i,13],liteDF.iat[i,14])#km
    firstSeen=datetime.strptime('+'.join(liteDF.iat[i,7].split('+', 1)[:1]),'%Y-%m-%d %H:%M:%S')
    lastSeen=datetime.strptime('+'.join(liteDF.iat[i,8].split('+', 1)[:1]),'%Y-%m-%d %H:%M:%S')
    travelledTime[i]=((lastSeen-dayOne).total_seconds()-(firstSeen-dayOne).total_seconds())/3600 #in hours
    averageVelocity[i]=distanceTravelled[i]/travelledTime[i] #in km/h
    averageAltitude[i]=(liteDF.iat[i,12]+liteDF.iat[i,15])/2*3.28084
    

plt.figure(2)
plt.title('distance travelled data (km)')
plt.scatter(np.random.randint(-50,50,size=(len(liteDF.altitude_1), 1)),distanceTravelled, alpha = 0.2)


plt.figure(3)
plt.title('travelled time data (hours)')
plt.scatter(np.random.randint(-50,50,size=(len(liteDF.altitude_1), 1)),travelledTime, alpha = 0.2)

plt.figure(4)
plt.title('average velocity data (km/h; ground speed)')
plt.scatter(np.random.randint(-50,50,size=(len(liteDF.altitude_1), 1)),averageVelocity, alpha = 0.2)

plt.figure(5)
plt.title('average velocity data (km/h; ground speed) vs. altitude (ft)')
plt.scatter(averageVelocity,(liteDF.altitude_1+liteDF.altitude_2)/2*3.28084, alpha = 0.2)

plt.figure(6)
plt.xlim(0,20)
plt.title('average flight time (hours) vs. altitude (ft)')
plt.scatter(travelledTime,(liteDF.altitude_1+liteDF.altitude_2)/2*3.28084, alpha = 0.2)


normalizedAltitude = [float(i)/max(averageAltitude) for i in averageAltitude]
normalizedVelocity = [float(i)/max(averageVelocity) for i in averageVelocity]

liteDF['avg Altitude']=averageAltitude
liteDF['avg Velocity']=averageVelocity
liteDF['distance travelled']=distanceTravelled
liteDF['normalized Velocity']=normalizedVelocity
liteDF['normalized Altitude']=normalizedAltitude

km=KMeans(n_clusters=5)
predictedKMclusters=km.fit_predict(liteDF[['normalized Velocity','normalized Altitude']])
liteDF['cluster']=predictedKMclusters


df1=liteDF[liteDF.cluster==0]
df2=liteDF[liteDF.cluster==1]
df3=liteDF[liteDF.cluster==2]
df4=liteDF[liteDF.cluster==3]
df5=liteDF[liteDF.cluster==4]
#df6=liteDF[liteDF.cluster==5]
#df7=liteDF[liteDF.cluster==6]

plt.figure(7) #cluster plot P vs. T
plt.title('average velocity vs. altitude (ft)')
plt.scatter(df1['avg Velocity'],df1['avg Altitude'], color='green',label='cluster1',alpha = 0.2)
plt.scatter(df2['avg Velocity'],df2['avg Altitude'], color='blue',label='cluster2',alpha = 0.2)
plt.scatter(df3['avg Velocity'],df3['avg Altitude'], color='red',label='cluster3',alpha = 0.2)
plt.scatter(df4['avg Velocity'],df4['avg Altitude'], color='orange',label='cluster4',alpha = 0.2)
plt.scatter(df5['avg Velocity'],df5['avg Altitude'], color='yellow',label='cluster5',alpha = 0.2)
#plt.scatter(df6['avg Velocity'],df6['avg Altitude'], color='black',alpha = 0.2)
#plt.scatter(df7['avg Velocity'],df7['avg Altitude'], color='yellow',alpha = 0.2)
plt.legend()
plt.show()

n=7 #number of most common values to show
print("cluster 1's common aircraft are",df1['typecode'].value_counts()[:n].index.tolist())
print("cluster 2's common aircraft are",df2['typecode'].value_counts()[:n].index.tolist())
print("cluster 3's common aircraft are",df3['typecode'].value_counts()[:n].index.tolist())
print("cluster 4's common aircraft are",df4['typecode'].value_counts()[:n].index.tolist())
print("cluster 5's common aircraft are",df5['typecode'].value_counts()[:n].index.tolist())