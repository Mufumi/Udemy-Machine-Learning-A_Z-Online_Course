#%%
#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header=None)
#"header=0" ensures that first row of data is not regarded as labels to data

#%%
#Creating applicable dataset for Apriori algorithm (List of lists)
transactions=[]
for i in range (0,len(dataset)):
        transactions.append([str(dataset.values[i,j]) for j in range(0,20)])
        #This line adds entries column by column (need to understand algorithm)
#%%
#Training the Apriori algorithm
from apyori import apriori
#apriori takes transactions(list of lists) as input and gives us rules as outputs
rules=apriori(transactions,min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)
# Arguments are support (will be dependent on the no. of transactions), confidence and lift
# min_length defines the minimum number of products that will be recommended

# min_support (we need to look at products purchased frequently), for this example, we look at products purchased 3-4 times a day
# Require at least 3 items per day (21 items per week)
# Therefore min_support=21/7500=0.0028

# min_confidence was set to 0.2. See discussion in ML Notes
# min_lift was set to 3 (to obtain relevant rules)
#%% Visualizing the results
results= list(rules)
#Relevance of the rules is dependent on support, confidence and lift (unlike R where criteria is based on lift)
