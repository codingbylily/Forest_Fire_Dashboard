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
import pickle

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

#link it to personal website
st.subheader( 'Another app brought to you by Lily from [CodingByLily](https://codingbylily.com/)')


#creating title
st.title('Forest Fire AI Prediction Dashboard')

st.write('Use the sidebar to input values from around your area of choice to see if there is a possibility of a forest fire occurring. Results will be displayed below.')

#plotting meteorological stations onto map of Portugal
meteorological_station = pd.read_csv('Portuguese meteorological stations.csv', usecols=['Name','Latitude (decimal degrees)', 'Longitude (decimal degrees)'])
meteorological_station.columns = ['Meteorological Station Name', 'latitude','longitude']
st.subheader('Mapped Locations of Meteorological Stations in Portugal')
st.map(meteorological_station, zoom=7, use_container_width=True)
st.write('Data from meteorological stations was used to train the machine learning model rather than satellite imaging and sensors such as fire alarms. Meteorological stations were found to have the least amount of delay in data when compared to satellites and sensors. Preventing delayed information will lead to faster response times for forest fire evacuation.')

#import meteorological stations dataset
meteorological_data = pd.read_csv('Portuguese meteorological stations.csv')
st.subheader('Portuguese meteorological station data')
meteorological_data

#header for forestfires dataset
st.subheader('Data collected from meteorological stations')
data
st.write('Data gathered from meteorological stations location in Montesinho park was used to predict the possibility of a forest fire occuring.')


# drawing a line chart for FFMC and temperature comparison within personal data
#chart_data = data[['temp','FFMC']]
#st.line_chart(chart_data)
# data.head()


#scatterplot
#fig = px.scatter(chart_data, x="temp", y="FFMC")
#st.plotly_chart(fig)



with open("model.pkl", "rb") as f:
    model = pickle.load(f)


#getting user inputs
ISI = st.sidebar.slider('Initial Spread Index (ISI)', min_value=0,max_value=56,value=4)
temp = st.sidebar.slider('Temperature (Celsius degrees)', min_value=2,max_value=33,value=4)
wind = st.sidebar.slider('Wind Speed (km/hr)', min_value=1,max_value=9,value=4)
rain = st.sidebar.slider('Outside Rain(mm/m^2)', min_value=0,max_value=6,value=0)
FFMC = st.sidebar.slider('Fine Fuel Moisture Code (FFMC)', min_value=19,max_value=96,value=78)

input_data = {'temp':[temp],

            'ISI':[ISI],

            'wind':[wind],

            'rain':[rain],
            'FFMC':[FFMC],
}



# input_data = [temp,ISI,wind,rain,FFMC]
# in_df = pd.DataFrame(data=input_data)


output_prediction = model.predict([[ISI,FFMC,wind,temp,rain]])

if output_prediction == 1:
    st.header(':red[Severe forest fire predicted]')
if output_prediction == 0:
    st.write('Severe forest fire NOT predicted')
st.write(output_prediction)





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

# x = data[['ISI','FFMC','wind','temp','rain']]
# y = data[['sev_index']]

# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

# clf = RandomForestClassifier(max_depth=2,random_state=0)

# clf.fit(X_train, y_train.values.ravel())


# y_predict = clf.predict(X_test)

# accuracy_score(y_test,y_predict)

# plt.style.use('seaborn-whitegrid')

# data.hist(bins=20, figsize=(14,10), color='#5D3FD3')
# plt.show()

#saving the model to disk?
#filename = 'finalized_model_clf.sav'
#joblib.dump(clf,filename)

# #loading the model from disk
#loaded_model = joblib.load(filename)
#result = loaded_model.score(X_test, y_test)
#print(result)



