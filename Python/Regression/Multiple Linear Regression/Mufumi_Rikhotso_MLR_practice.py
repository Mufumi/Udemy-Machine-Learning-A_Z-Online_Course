# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Apply data pre-processing from the template

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values #Cannot visualize from variable explorer. Can view when typing on console
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#NB The dependent variable is not categorical,therefore it does not need to be encoded

#Avoiding the Dumy variable trap

X=X[:,1:] #Slicing the X array from the 1st column and thus reducing the number of dummy variables


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Fitting Multiple Linear Regression to Training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train,y_train)

# We are now going to test the performance of the model

y_pred=regressor.predict(X_test)

import statsmodels.formula.api as sm

#In Multivariable linear regression model, b_0 has coefficient x_0=1 but this library does not consider that
#We have to introduce this part of the equation
#To do this, we have to ADD column of 1s to the X matrix

X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)
#This line appends the matrix X at the end of a column of 1s
#np.ones creates a column of ones where the size is specified with the type specified
#axis shows that it is a column

"""Backward Elimination"""

X_optimal= X[:,[0, 1, 2, 3, 4, 5]]#This matrix has variables(independent) that have the highest weight on the dependent variable
#This matrix was created like this so that we can apply column deletion later (when eliminating predictors with little wight)
"""STEP 1: Select Significance Level"""

sl=0.05

"""STEP 2: Fit model with all possible predictors"""
regressor_OLS=sm.OLS(endog=y,exog=X_optimal).fit()
#In the regressor_OLS object. We want to apply the ordinary least squares method to the object and fit it to the regressor_OLS object

"""STEP 3: Consider predictor with highest p-value if p-value>SL go to step 4, else fin"""
regressor_OLS.summary()
#Gives us a summary of statistical variables that can be used to determine p-value
#The lower the p-vlaue, the more it affects the dependent variable
#Const is the 1s vector we introduced
#x1-Dummy variable 1, x2-Dummy variable 2, x3-R&D Spend, x4- Administration, x5- Marketing spend


"""STEP 4: Remove predictor"""

#x2 has the highest p-value therefore we have to remove it and fit the model again and determine the p-values again
#X_optimal= X[:,[0, 1, 3, 4, 5]]

#x1 has the highest p-value therefore we have to remove it and fit the model again and determine the p-values again
#X_optimal= X[:,[0, 3, 4, 5]]

#x4 has highest p-value therefore we have to remove it and fit the model again and determine the p-values again
#X_optimal= X[:,[0, 3, 5]]

#x5 has highest p-value therefore we have to remove it and fit the model again and determine the p-values again
#X_optimal= X[:,[0, 3]]

"""STEP 5: Fit model without this predictor"""

""" From iterative process, we find that the R&D is the only value whose p-value<SL"""

########################################Backward Elimination with p-values only#############################################
"""
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
"""

########################################Backward Elimination with p-values and Adjusted R Squared#############################################
"""
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
"""
