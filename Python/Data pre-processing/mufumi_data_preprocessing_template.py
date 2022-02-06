# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename='Data.csv'

dataset= pd.read_csv(filename)

#X = dataset.iloc[:, :-1].values #slicing technique

"""
We are sling all the rows using the first ":" and slicing all the columns besides the last one (decision vector) using the second ":"
.values is just all the values
Python already loads the labels and does not include them in the dataset
"""
# We are going to create the dependant variable vector

#y= dataset.iloc[:,3].values
