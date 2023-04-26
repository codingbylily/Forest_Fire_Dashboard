import sklearn
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib as plt
#import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


import streamlit as st 


# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title="Forest Fire Dashboard", page_icon=":fire:")

#import joblib
import joblib

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
#import our dataset and print the first 5 rows
data = pd.read_csv('forestfires.csv')

data

# data.head()

# #explore last 5 rows
# data.tail()

# data.describe()
# data.info()
# data['log_area'] = np.log10(data['area']+1)
# data.head()
# data.describe()

# #plt.pyplot.scatter(data['X'], data['Y'],s = 20, c=data['log_area'], cmap='spring')
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

# #plt.style.use('seaborn-whitegrid')

# data.hist(bins=20, figsize=(14,10), color='#5D3FD3')
# #plt.show()

# #saving the model to disk?
# filename = 'finalized_model_clf.sav'
# joblib.dump(clf,filename)

# #loading the model from disk
# loaded_model = joblib.load(filename)
# result = loaded_model.score(X_test, y_test)
# print(result)


# #testing streamlit
st.write('hello world')
