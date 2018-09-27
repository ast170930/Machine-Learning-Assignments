# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 17:10:06 2018

@author: Abdramane
"""

import numpy as np                                       
import pandas as pd                                      
from sklearn.model_selection import train_test_split     
from sklearn.tree import DecisionTreeRegressor 

wrdsdata = pd.read_csv("C:/Users/Abdramane/Desktop/UTD Classes/Semester 3 - Fall 2018/2 - Machine Learning for Financial Applications/compustat_annual_2000_2017_with link information.csv", low_memory = False)

# Show number of columns (variables) and rows (observations)
wrdsdata.shape

# Show total number of null values in each column of the dataset 
wrdsdata.isnull().sum()

# Remove and store columns with less than 70% missing values
df = wrdsdata[wrdsdata.columns[wrdsdata.isnull().sum() / wrdsdata.shape[0] < 0.7]]
df

# Remove columns from new dataframe with non-numeric data
df.select_dtypes([np.number])

# Show statistical summary of numerical variables
dfq = df.select_dtypes([np.number])
dfq.describe()

# Check for missing values in the dataset
dfq.isnull().sum()

# Fill in the missing values in the dataset
dfq.fillna(dfq.median())

# Check for missing values for the new dataset
dfw = dfq.fillna(dfq.median())
dfw.isnull().sum()

dfw['ebit']

X = dfw[dfw.columns[~dfw.columns.isin(['ebit'])]]    # All other variables except ebit, independent variables (predictor)
Y = dfw['ebit']                                      # Ebit as the dependent variable
max_depth1 = 4                                       # Maximum number of nodes for decision tree one
max_depth2 = 5                                       # Maximum number of nodes for decision tree two
min_size = 10                                        # Minimum number of training patterns

# Cross-validation to evaluate algorithm and training test/split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Tree depth of 4
tree1 = DecisionTreeRegressor(random_state = 0, max_depth = 4, min_samples_leaf = 10)
tree1.fit(X_train, Y_train)

Y_m1 = tree1.predict(X_test)
Y_m1

# Tree depth of 5
tree2 = DecisionTreeRegressor(random_state = 0, max_depth = 5, min_samples_leaf = 10)
tree2.fit(X_train, Y_train)

Y_m2 = tree2.predict(X_test)
Y_m2