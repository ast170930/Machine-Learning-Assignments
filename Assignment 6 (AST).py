# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 00:03:41 2018

@author: Abdramane
"""

import numpy as np                                       # Scientific computing and arrays
import pandas as pd                                      # Data structure ana analysis tools
from sklearn.ensemble import RandomForestRegressor       # Import random forest from Scikit
from sklearn.ensemble import RandomForestClassifier      # Import random forest model
from sklearn.metrics import mean_squared_error as mse    # Import MSE from scikit
from sklearn.model_selection import train_test_split     # For random train and test splits
from sklearn.preprocessing import StandardScaler         # For feature standardization
from keras.models import Sequential                      # Configure the model for training
from keras.layers import Dense   

#Putting cds data in a dataframe
cdsdata = pd.read_stata("C:/Users/Abdramane/Documents/cds_spread5y_2001_2016.dta")

#Checking the shape of cdsdata
cdsdata.shape

#Checking types of cdsdata
cdsdata.dtypes

#Showing the 5 first rows of cdsdata
cdsdata.head()

#Putting the quartely compustat data in dataframe
qmcdata = pd.read_csv("C:/Users/Abdramane/Documents/Quarterly Merged CRSP-Compustat.csv", low_memory = False)

#Checking the shape of qmcdata
qmcdata.shape

#Checking types of qmcdata
qmcdata.dtypes

#Showing the 5 first rows of qmcdata
qmcdata.head()

#Separating date, month, and year in cdsdata
cdsdata['Date'] = pd.to_datetime(cdsdata['mdate'])
cdsdata['Month'] = cdsdata['Date'].dt.month
cdsdata['Year'] = cdsdata['Date'].dt.year

#Setting the different quarters in cdsdata
cdsdata['Quarter'] = '4'
#For quarter 4
cdsdata['Quarter'][cdsdata['Month'] > 9] = '4'
#For quarter 3
cdsdata['Quarter'][(cdsdata['Month'] > 6) & (cdsdata['Month'] < 9)] = '3'
#For quater 2
cdsdata['Quarter'][(cdsdata['Month'] > 3) & (cdsdata['Month'] < 6)] = '2'
#For quater 1
cdsdata['Quarter'][cdsdata['Month'] < 3] = '1'

#Transforming the columns in cdsdata to float
cdsdata['gvkey'] = cdsdata['gvkey'].astype(float)
cdsdata['Quarter'] = cdsdata['Quarter'].astype(float)
cdsdata['Year'] = cdsdata['Year'].astype(float)

#Changing columns names in qmcdata to match colums in cdsdata
qmcdata = qmcdata.rename(columns = {'datadate':'mdate'})
qmcdata = qmcdata.rename(columns = {'GVKEY':'gvkey'})

#Separating date, month, and year in qmcdata
qmcdata['Date'] = pd.to_datetime(qmcdata['mdate'])
qmcdata['Month'] = qmcdata['Date'].dt.month
qmcdata['Year'] = qmcdata['Date'].dt.year

#Setting the different quarters in qmcdata
qmcdata['Quarter'] = '4'
#For quarter 4
qmcdata['Quarter'][qmcdata['Month'] > 9] = '4'
#For quarter 3
qmcdata['Quarter'][(qmcdata['Month'] > 6) & (qmcdata['Month'] < 9)] = '3'
#For quater 2
qmcdata['Quarter'][(qmcdata['Month'] > 3) & (qmcdata['Month'] < 6)] = '2'
#For quater 1
qmcdata['Quarter'][qmcdata['Month'] < 3] = '1'

#Transforming the columns in qmcdata to float
qmcdata['gvkey'] = qmcdata['gvkey'].astype(float)
qmcdata['Quarter'] = qmcdata['Quarter'].astype(float)
qmcdata['Year'] = qmcdata['Year'].astype(float)

#Merging cdsdata and qmcdata
newdata = pd.merge(qmcdata, cdsdata, on=['gvkey', 'Quarter','Year'])

#Counting the null values in each dataframe columns 
newdata.isnull().sum()

newdata.describe()

#Checking for missing values
newdata.fillna(newdata.median())

#Keep only numerical variables
newdata1 = newdata.select_dtypes([np.number])
newdata1

#Checking for missing values
newdata1.isnull().sum()

#Remove variables that are all missing
newdata2 = newdata1.dropna(axis=1, how='any')

#Split data
X = newdata2.drop('spread5y', axis=1)
y = newdata2['spread5y']

#Split into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Remove other features in test data
X_test = X_test.drop('Month_x', axis=1)
X_test = X_test.drop('Month_y', axis=1)
X_test = X_test.drop('Quarter', axis=1)
X_test = X_test.drop('Year', axis=1)
X_test = X_test.drop('gvkey', axis=1)

#Remove other features in train data
X_train= X_train.drop('Month_x', axis=1)
X_train= X_train.drop('Month_y', axis=1)
X_train= X_train.drop('Quarter', axis=1)
X_train= X_train.drop('Year', axis=1)
X_train= X_train.drop('gvkey', axis=1)

#Standardizing features
scaler = StandardScaler()
#Fit only to the training data
scaler.fit(X_train)
StandardScaler(copy = True, with_mean = True, with_std = True)

X_train_1 = X_train
X_test_1 = X_test

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Random forest classifier
#Number of trees = 10
rf = RandomForestRegressor(n_estimators = 10) 
rf.fit(X_train, y_train) 

#Score the model
rf.score(X_test, y_test)
rf40 = rf.predict(X_test)

#Segragate top 40 features
feature_importances = rf.feature_importances_
feature_importances = pd.DataFrame(rf.feature_importances_, 
                                   index = X_train_1.columns, 
                                    columns = ['importance']).sort_values('importance',ascending = False)
top40 = feature_importances.iloc[:40, :]
top40 = top40.index.tolist()

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(y_test, rf40)

#Filter top 40 features
Fil_X_train = X_train_1[top40]
Fil_X_test = X_test_1[top40]

# Filter the fitted standardized training data
scaler.fit(Fil_X_train)
StandardScaler(copy = True, with_mean = True, with_std = True)

Fil_X_train_1 = scaler.transform(Fil_X_train)
Fil_X_test_1 = scaler.transform(Fil_X_test)

#Neural Network
#Generate fix random seed
np.random.seed(7)
#Create multilayer perceptron (MLP) model
model = Sequential()
model.add(Dense(32, input_dim = 18, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'linear'))
#Compile MLP
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
#Fit MLP
model.fit(Fil_X_train_1, y_train, epochs = 100, batch_size = 10)

#Predicting the first model
M1 = model.predict(Fil_X_test_1)
mapeM1 = mean_absolute_percentage_error(y_test,M1)
mapeM1

#Create second model
model = Sequential()
model.add(Dense(32, input_dim = 40, activation = 'sigmoid'))
model.add(Dense(8, activation = 'sigmoid'))
model.add(Dense(1, activation = 'relu'))
#Compile MLP
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
#Fit MLP
model.fit(Fil_X_train_1, y_train, epochs = 30, batch_size = 10)
#Predicting the second model
M2 = model.predict(Fil_X_test_1)
mapeM2 = mean_absolute_percentage_error(y_test, M2)
mapeM2

#Create third model
model = Sequential()
model.add(Dense(32, input_dim = 40, activation = 'hard_sigmoid'))
model.add(Dense(8, activation = 'elu'))
model.add(Dense(1, activation = 'relu'))
#Compile MLP
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
#Fit MLP
model.fit(Fil_X_train_1, y_train, epochs = 30, batch_size = 10)
#Predicting the third model
M3 = model.predict(Fil_X_test_1)
mapeM3 = mean_absolute_percentage_error(y_test, M3)
mapeM3

print("Mean Absolute Percentage for M model 1:  ",mapeM1)
print("Mean Absolute Percentage for M model 2:  ",mapeM2)
print("Mean Absolute Percentage for M model 3:  ",mapeM3)