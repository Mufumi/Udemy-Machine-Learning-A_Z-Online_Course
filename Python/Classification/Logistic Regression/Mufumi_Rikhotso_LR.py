#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 10:48:31 2019

@author: user
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv('Social_Network_Ads.csv')
# We want to predict whether or not the age and salary affect the choice of purchase

X = dataset.iloc[:,[2,3]] 
y=dataset.iloc[:,4]

# Splitting the dataset into Training set and Testing set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
# Feature scaling is applied (usually because probability will be between 0 and 1)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Fiiting Logistic Regression to Training set

from sklearn.linear_model import LogisticRegression
classifier= LogisticRegression(random_state=0) #only use random state parameter
classifier.fit(X_train,y_train)

#Predicting the Test results
y_pred=classifier.predict(X_test)

#Making the Confussion Matrix
from sklearn.metrics import confusion_matrix #This is a function instead of a class
cm=confusion_matrix(y_test,y_pred) #Print on terminal

#Visualizing the Training set results
from matplotlib.colors import ListedColormap #Enables ability to colorize
X_set, y_set = X_train, y_train #Variables to avoid repition of X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#resolution is important as it makes it seems as if the regions are fine
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#This is the most important part of the visualization, it predicts every pixel 
plt.xlim(X1.min(), X1.max()) #Plot the limits
plt.ylim(X2.min(), X2.max()) #Plot the limits
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#Visualizing the Test set
"""
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
"""