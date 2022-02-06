# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
# Encoding the Independent Variable

# Label encoder is the class we use for labelling categorical data

from sklearn.preprocessing import LabelEncoder#, OneHotEncoder
labelencoder_X = LabelEncoder() #Creating LabelEncoder object 
X[:, 0] = labelencoder_X.fit_transform(X[:, 0]) # Here we implement a method of the LabelEncoder class
# The encoding will only be done for the first column

"""One problem that arises from this econding is that when comparing the countries, the
computer ranks the encoded countries by orders of magnitude i.e Spain is geater than France,
however that doesn't make sense"""

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)