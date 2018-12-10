
# coding: utf-8

# # Simple Linear Regression on MBTA Orange Line
# 
# **Assignment Due Date:** End of Day Friday November 30th
# 
# **Submission:** Submit just this notebook to James through email by the above due date. You can also share the homework via email using an online storage (e.g. Box, Google Drive)
# 
# Build a simple linear regression model for the MBTA subway Orange Line.  Your model should have the following constraints:
# - 1)Use both September and October historical data for Orange Line to build model ('mbta_Orange_09_2018.json' and 'mbta_Orange_10_2018.json')
# - 2)Regression model is for the direction of train moving from Forest Hills to Oak Grove (opposite direction should be removed)
# - 3)Regression model should be for trips that occur on Saturday and only between 7am - 10pm (all other days and time outside the specified range should not be included in the regression model)
# - 4)Unique trips for a specific day that have under 40 vehicle updates should removed
# - 5)Unique trips for a specific day in which the vehicle updates do not begin at Forest Hills should be removed
# - 6)The dependent variable should be the elapsed time since trip's began and should be represented in 'hour' unit 
# - 7)The independent variable should be distance traveled from start and should be represented in 'kilometer' unit
# 
# Please tag the portion of your code that handles each of the above constraints with '#CONSTRAINT{bullet-number}.'  For example if you are filtering out any trips that do not occur on Saturday.  Prior to the logic that performs this put a comment '#CONSTRAINT3.'  If you have logic for a specific constraint spread throughout the notebook please tag each piece. 
# 
# Plot your simple linear regression model and include a scatter plot for the testing and training dataset.
# 
# You are encouraged to reuse the logic from the lectures to complete this assignment.  The historical datasets used to build the regression model can be found here:  https://umass.app.box.com/s/x3zgwv34uduqrxnwkako4rbkjayzabt1/folder/56231379908 . 
# 
# Full credit given to if entire work is shown and follows the above constraints.  This homework is an individual assignment.  The code will be re-run locally and that runtime output is what will be graded, not the output displayed when submitted. **Please be sure your code runs as expected from the start of the notebook to the end.**

# ![MBTAOrangeLineHW3.png](attachment:MBTAOrangeLineHW3.png)
# 
# Example plot of Orange Line to Oak Groves distance traveled vs time elapsed:
# 
# ![forestHillsOak.png](attachment:forestHillsOak.png)
# 
# Example plot of simple linear regression model with scatter plot of testing and training dataset:
# 
# ![exampleOrangeRegression.png](attachment:exampleOrangeRegression.png)

# In[1]:


import json
import pandas as pd
import requests
import numpy as np
from geopy import distance
import plotly as py
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


py.offline.init_notebook_mode(connected=True)


# In[3]:


##CONSTRAINT1
# Using Orange Line Data from September as well as October 2018
with open('mbta_Orange_09_2018.json') as file:
    SeptData = json.load(file)
with open('mbta_Orange_10_2018.json') as file:
    OctData = json.load(file)
SeptDF = pd.io.json.json_normalize(SeptData, record_path = "Vehicles")
OctDF = pd.io.json.json_normalize(OctData, record_path = "Vehicles")
FinalDF = pd.concat([SeptDF, OctDF])


# In[4]:


# Clearing out and droping unwanted columns
DropList = "Speed Type Bearing Label RouteId".split()
FinalDF1 = FinalDF.drop(DropList, axis = 1)
FinalDF1 = FinalDF1.sort_values(["Id","UpdatedAt"])


# In[5]:


#CONSTRAINT2
#Neglecting the journy from Oak Grove to Forest Hills
FinalDF1 = FinalDF1[FinalDF1["DirectionId"]==1] #Forest Hills to Oak Grove


# In[6]:


FinalDF1['UpdatedAt'] = pd.to_datetime(FinalDF1['UpdatedAt'])
FinalDF1['Day'] = FinalDF1['UpdatedAt'].dt.dayofyear
FinalDF1['Hour'] = FinalDF1['UpdatedAt'].dt.hour
FinalDF1['Saturday'] = FinalDF1['UpdatedAt'].dt.dayofweek==5


# In[7]:


#CONSTRAINT3
# Taking trips between 7AM and 10PM on Saturdays
FinalDF1 = FinalDF1.query('Saturday==True')
FinalDF1 = FinalDF1.query('Hour>=7 & Hour<=22')


# In[8]:


#CONSTRAINT4
#Removing Trips with less than 40 updates
group = ['TripId', 'Day']
FinalDF2 = FinalDF1.groupby(group).filter(lambda x: x['CurrentStatus'].count()>40)


# In[9]:


#CONSTRAINT5
#Removing Specially Added Trips
FinalDF2 = FinalDF2[~FinalDF2['TripId'].str.contains('ADDED-')]


# In[10]:


#This section deals with merging information from historic data with the data available on the Web API. 
#We are merging Stop Names from a data available on the web with historic data and evaluating Distance between stops

url = "https://api-v3.mbta.com/stops?page%5Boffset%5D=0&page%5Blimit%5D=100&filter%5Bdirection_id%5D=1&filter%5Broute%5D=Orange"
r = requests.get(url)
OrangeStops = json.loads(r.content)
for stop in OrangeStops['data']:
    stop['latitude'] = stop['attributes']['latitude']
    stop['longitude'] = stop['attributes']['longitude']
    stop['name'] = stop['attributes']['name']
    stop.pop('links')
    stop.pop('attributes')
    stop.pop('relationships')
    

data = json.dumps(OrangeStops['data'])
StopDF = pd.read_json(data)
sIDDF = pd.read_csv('stops.txt')
sIDDF = sIDDF[sIDDF['stop_code'].notnull()]
OrangeStopsDF = pd.DataFrame(FinalDF2['StopId'].unique())
OrangeStopsDF = OrangeStopsDF.merge(sIDDF, left_on = 0, right_on = 'stop_id')
StopDF = StopDF.merge(OrangeStopsDF, left_on = 'name', right_on = 'stop_name')
Select = ['stop_id', 'latitude', 'longitude','name', 'stop_name','platform_name','stop_lat', 'stop_lon', 'level_id', 'parent_station', 'wheelchair_boarding']
StopDF = StopDF[Select].copy()
StopDF['PastCoordinate'] = StopDF['stop_lat'].shift(1).astype('str').str.cat(StopDF['stop_lon'].shift(1).astype('str'), sep=',')

def FixNaNVal(df):
    if(df['PastCoordinate']=='nan,nan'):
        return '{},{}'.format(df['stop_lat'],df['stop_lon'])
    return df['PastCoordinate']

StopDF['PastCoordinate'] = StopDF.apply(lambda x: FixNaNVal(x), axis = 1) 


# In[11]:


#CONSTRAINT7
#Representing Distance in Km
StopDF['DistanceFromPrior'] = StopDF.apply(lambda x: distance.distance(x['PastCoordinate'],'{},{}'.format(x['stop_lat'],x['stop_lon'])).km, axis = 1)
StopDF['DistanceFromOrigin'] = StopDF['DistanceFromPrior'].cumsum()
StopDF.drop(['PastCoordinate','DistanceFromPrior'], axis = 1, inplace = True)


# In[12]:


#Evaluating distance the train has moved with the progress in time

Merge = StopDF[['stop_id','name','DistanceFromOrigin','latitude','longitude']].copy()
Merge['StopLatLong'] = Merge['latitude'].astype('str').str.cat(Merge['longitude'].astype('str'), sep = ',')
Merge = Merge.drop(['latitude','longitude'], axis = 1)
FinalDF2 = FinalDF2.merge(Merge, left_on='StopId', right_on='stop_id')

def DistanceFromStop(df):
    if(df['CurrentStatus']=='STOPPED_AT'):
        return 0
    return distance.distance(df['StopLatLong'],'{},{}'.format(df['Latitude'],df['Longitude'])).km

FinalDF2['DistanceFromStop'] = FinalDF2.apply(lambda x: DistanceFromStop(x), axis = 1)
FinalDF2['StopDistance'] = FinalDF2['DistanceFromOrigin']
FinalDF2['DistanceFromOrigin'] = FinalDF2['StopDistance']-FinalDF2['DistanceFromStop']
Group1 = ['TripId','Day']
FinalDF2['Elapsed'] = FinalDF2.groupby(Group1)['UpdatedAt'].transform(lambda x: x-x.min())


# In[13]:


def CreateTripTrace(distance, time, name, line = 0.05):
    return go.Scattergl(
        x = time,
        y = distance,
        name = name,
        mode = 'lines',
        line = dict(
            color = ('rgb(5,40,205)'),
            width = line,
        )
    
    )


# In[18]:


#CONSTRAINT6
#Representing the time elapsed during a trip in hours

stops = StopDF[['name','DistanceFromOrigin']]

def CreateStopTrace(distance, name):
    return go.Scattergl(
        y = (distance,distance),
        x = [0, 0.8],
        name = name,
        mode = 'lines',
        line = dict(
            color = ('rgb(255,40,0)'),
            width = 0.5
        )
    )

def CreateStopAnn(StopTrace):
    return dict (xref = 'paper', x = 0.85, y = StopTrace['y'][0], xanchor = 'left', yanchor = 'bottom', text = StopTrace['name'],
                showarrow = False)


def CreateStopPlots(TraceData, stops, annotations):
    temp = stops.apply(lambda x: CreateStopTrace(x['DistanceFromOrigin'], x['name']), axis = 1)
    StopTraces = list(temp.values)
    TraceData.extend(StopTraces)
    for stopTrace in StopTraces:
        annotationtrace = CreateStopAnn(stopTrace)
        annotations.append(annotationtrace)
        
def DistVsTime(StopsDF, UpdatesDF, max_display = -1, LinwWidth = 0.5):
    data = []
    annotations = []
    count = 1
    CreateStopPlots(data, StopDF, annotations)

    for name, group in UpdatesDF.groupby(Group1):
        HourList = group.apply(lambda x: x['Elapsed'].total_seconds()/3600, axis = 1) #Converting Into Hours
        trace = CreateTripTrace(group['DistanceFromOrigin'], HourList, '{}'.format(name), LinwWidth)
        data.append(trace)
        if(count == max_display):
            break
        count+=1


    layout = dict(title = 'Trips on Orange Line',
                  xaxis = dict(title = 'Elapsed Time In Hours'),
                  yaxis = dict(title = 'Distance Travelled in Km'),
                  showlegend = False,
                  annotations = annotations
                 )
    figure = dict(data = data, layout = layout)
    py.offline.iplot(figure)
DepartDF = FinalDF2[FinalDF2['name'] != 'Forest Hills'].copy()
DepartDF['Elapsed'] =  DepartDF.groupby(Group1)['UpdatedAt'].transform(lambda x: x-x.min())
DistVsTime(stops, DepartDF, 100, 1)


# In[15]:


# Preparing a linear regression model with the filtered data set:

DepartDF['Elapsed'] = pd.to_timedelta(DepartDF['Elapsed']).dt.total_seconds()/3600 
DepartDF1 = DepartDF.drop(['StopLatLong', 'DistanceFromStop', 'StopDistance'], axis = 1)
Attribute = Target = DepartDF1.iloc[:,-2:-1].values
Target = DepartDF1.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(Attribute, Target, test_size = 0.2, random_state = 0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[16]:


# Linear Regression Model Plot

def PlotLinearRegression(x_train,y_train,x_test,y_test,regressor):
    tracetrain = go.Scattergl(
        y = y_train.ravel(),
        x = x_train.ravel(),
        name = 'training Points',
        mode = 'markers'
    )
    tracetest = go.Scattergl(
        y = y_test.ravel(),
        x = x_test.ravel(),
        name = 'training Points',
        mode = 'markers'
    )
    tracepred = go.Scattergl(
        y = regressor.predict(x_train).ravel(),
        x = x_train.ravel(),
        name = 'predicted line',
        mode = 'lines+markers'
    )
    layout = dict(title = 'Trips on Orange Line',
                  xaxis = dict(title = 'Distance Travelled in Km'),
                  yaxis = dict(title = 'Elapsed Time In Hours'),
                  showlegend = True,
                 )
    data = (tracetrain, tracetest, tracepred)
    figure = dict(data = data, layout = layout)
    py.offline.iplot(figure)
    
PlotLinearRegression(x_train,y_train,x_test,y_test,regressor)


# In[17]:


# RMS Error evaluation

result = regressor.predict(x_test)
length = len(result)
RMSE = (np.sqrt(sum((y_test-result)*(y_test-result))/length))*100
print("Root Mean Square Error (%)")
print(RMSE)

