# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 19:09:37 2018

@author: Abdramane
"""

import csv
import pandas as pd
import numpy as np
import statsmodels.api as sm

#To read the csv file  
dataset = pd.read_csv("C:/Users/Abdramane/Desktop/UTD Classes/Semester 3 - Fall 2018/2 - Machine Learning for Financial Applications/compustat_annual_2000_2017_with link information.csv", low_memory = False)

#Number of rows and columns 
dataset.shape

dataset.head()

#Counting the null values in each dataframe columns 
dataset.isnull().sum()

#Statistics for numerical variables
dataset.describe()

npset = dataset.select_dtypes(include = np.number)

#Keeping columns with less than 70% missing values
dataset_kpcols = npset[npset.columns[npset.isnull().sum()/npset.shape[0]<0.7]]

dataset_kpcols.head()

#Fill in the median
dataset_kpcols.fillna(dataset_kpcols.median())

#Check for missing values for the new imputed dataset
rex = dataset_kpcols.fillna(dataset_kpcols.median())
rex.apply(lambda x: sum(x.isnull()), axis = 0)

#Define ("ebit", dataset_kpcols.ebit)
X = dataset_kpcols[dataset_kpcols.columns[~dataset_kpcols.columns.isin(['ebit'])]]
Y = dataset_kpcols[dataset_kpcols.columns[dataset_kpcols.columns.isin(['ebit'])]]

def stepwise_regression(X, Y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    included = list(initial_list)
    while True:
        changed=False
# Using "forward step"
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
# Using "backward step"
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

result = stepwise_regression(X, Y)
print(result)