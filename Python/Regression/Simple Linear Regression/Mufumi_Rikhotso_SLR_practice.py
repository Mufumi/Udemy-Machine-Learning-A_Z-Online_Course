#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:57:42 2018

@author: user
"""
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('50_Startups.csv')

X=dataset.iloc[:,:3].values #'.values' ensures that the entries are values of the numpy.array

y=dataset.iloc[:,-1].values

#train_test_split

from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test=train_test_split()


from sklearn.linear_model import LinearRegression

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import data
dataset = pd.read_csv('Salary_Data.csv')

#Determine the indipendent variable vectors and dependent variable vector
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#Split data into train, test 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_predict=regressor.predict(X_test)

plt.plot(X_train, regressor.predict(X_train),color='blue')
#y=
#X_train, X_test, y_train, y_test=train_test_split