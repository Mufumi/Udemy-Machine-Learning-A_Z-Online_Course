# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 23:31:18 2018

@author: Mufumi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Must import dataset
dataset=pd.read_csv('Position_Salaries.csv')



X = dataset.iloc[:, 1:2].values #We only need the level as it will automatically tell of the position
#Always consider independent variables as a matrix

y = dataset.iloc[:, 2].values

#No need for feature scaling and train, test, split
#Linear regresssion model implements feature scaling

#Linear regresion and polynomial regression model will  be compared

from sklearn.linear_model import LinearRegression
Lin_reg= LinearRegression() #First object for simple linear regression
Lin_reg.fit(X,y)

#Fitting polynomial Regression to 

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) #Transforms matrix of features of x into x^2 and higher orders

#Create x_poly matrix

X_poly=poly_reg.fit_transform(X)
#This matrix creates a polynomial form of X
#

Lin_reg2=LinearRegression()
Lin_reg2.fit(X_poly,y)

#Visualize Linear Regression results

plt.scatter(X,y,color='red')
plt.plot(X,Lin_reg.predict(X),color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualize Polynomial Regression results

#If poly_reg is used, it gives an error. This is because it is still
#...part of the Linear Regression class. 
X_grid=np.arange(min(X),max(X), 0.1) #Creating numpy range
#This is to create a smoother graph
X_grid=X_grid.reshape((len(X_grid),1))
#Reshaping 

plt.scatter(X,y,color='red')
plt.plot(X_grid,Lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
#This line creates a high resolution graph with high level accuracy

plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#Predict result with Linear Regression

"""Lin_reg.predict(6.5)"""
#This line of code does not work

#Predict result with Polynomial Regression
"""Lin_reg2.predict(poly_reg.fit_transform(6.5))"""


#This line of code also does not work