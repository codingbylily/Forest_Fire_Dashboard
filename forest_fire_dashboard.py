import sklearn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
#import pickle

import streamlit as st 


# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title="Forest Fire Dashboard", page_icon=":fire:")

#import joblib
import joblib

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
#import our dataset
data = pd.read_csv('forestfires.csv')

#creating title
st.title('Forest Fire AI Prediction Dashboard')

#plotting meteorological stations onto map of Portugal
meteorological_station = pd.read_csv('Portuguese meteorological stations.csv', usecols=['Name','Latitude (decimal degrees)', 'Longitude (decimal degrees)'])
meteorological_station.columns = ['Meteorological Station Name', 'latitude','longitude']
st.subheader('Portuguese Meteorological Stations Locations')
st.map(meteorological_station, zoom=3, use_container_width=True)
st.caption(':red[We will be using data from meteorological stations rather than satellite or sensor data to train out machine learning model. Meteorological stations were found to have the least amount of delay in data when compared to satellite or sensor data. Preventing delayed information will lead to faster response times for forest fire evacuation.]')

#import meteorological stations dataset
meteorological_data = pd.read_csv('Portuguese meteorological stations.csv')
st.subheader('Portuguese meteorological stations data')
meteorological_data

#header for forestfires dataset
st.subheader('About the data behind the model')
data
st.caption(':red[Data gathered from meteorological stations location in Montesinho park.]')

drawing a line chart for FFMC and temperature comparison within personal data
chart_data = data[['temp','FFMC']]
st.line_chart(chart_data)
# data.head()


#scatterplot
fig = px.scatter(chart_data, x="temp", y="FFMC") 
st.plotly_chart(fig)


# #explore last 5 rows
# data.tail()

# data.describe()
# data.info()
# data['log_area'] = np.log10(data['area']+1)
# data.head()
# data.describe()

#plt.pyplot.scatter(data['X'], data['Y'],s = 20, c=data['log_area'], cmap='spring')
# #plt.pyplot.scatter(data['rain'],data['log_area'])
# #plt.pyplot.scatter(data['temp'],data['log_area'])
# #plt.pyplot.scatter(data['FFMC'],data['log_area'])
# #plt.pyplot.scatter(data['DMC'],data['log_area'])
# #plt.pyplot.scatter(data['wind'],data['log_area'])
# #plt.pyplot.scatter(data['ISI'],data['log_area'])

# def sev_val(row):
#     #Creates new column to indicate samples of interest
#     #We want the 
#     if row['area'] <2:
#         val = 1
#     else:
#         val = 0
#     return(val)
    
    
# data['sev_index'] = data.apply(sev_val, axis=1)

#x = data[['ISI','FFMC','wind','temp','rain']]
#y = data[['sev_index']]

#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

#clf = RandomForestClassifier(max_depth=2,random_state=0)

#clf.fit(X_train, y_train.values.ravel())

#y_predict = clf.predict(X_test)

#accuracy_score(y_test,y_predict)

#plt.style.use('seaborn-whitegrid')

data.hist(bins=20, figsize=(14,10), color='#5D3FD3')
plt.show()

#saving the model to disk?
#filename = 'finalized_model_clf.sav'
#joblib.dump(clf,filename)

# #loading the model from disk
#loaded_model = joblib.load(filename)
#result = loaded_model.score(X_test, y_test)
#print(result)



