# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #defining the feature matrix
y = dataset.iloc[:, 3].values #defining the decision vector

# Taking care of missing data
from sklearn.preprocessing import Imputer #This is the Imputer class 
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #Characteristics of the imputer we want to implement
""" For the imputer, we want to implement an imputer to replace the NaN variables with the mean of that specific column"""

imputer = imputer.fit(X[:, 1:3]) #This applies the fit method to column 1 and 2
X[:, 1:3] = imputer.transform(X[:, 1:3]) #Filling in the values of the missing variables