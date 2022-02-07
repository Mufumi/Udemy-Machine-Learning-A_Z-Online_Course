# %% This defines a new code cell

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
#There is no indication of the groups existing in the dataset
#Also, there is no idea of the number of groups in the data
# This makes it a clustering problem (There is no known categories ahead of time).

#%%
X=dataset.iloc[:,[3,4]].values

#%%
#Using dendogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(X,method='ward'))

#sch.linkage lists which data has to represented as a dendogram. Linkage is a function of hierarchical clustering
#method - method used to find the clusters. Ward method tries minimize the varience in the clusters. Similar to WCSS but instead of within cluster varience(varience in each cluster)

#Ploting the dendogram
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
#%%
#Fitting Hierarchcal clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
#Create object of Agglomerative clustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')

y_hc=hc.fit_predict(X)
#fit_predict returns vector of clusters

#%%
#Visualizing the clusters
#Note for multidimensional problem, this section is not applicable
plt.scatter(X[y_hc==0,0], X[y_hc==0,1],s=100, c = 'red', label = 'Careful')
plt.scatter(X[y_hc==1,0], X[y_hc==1,1],s=100, c = 'blue', label = 'Standard')
plt.scatter(X[y_hc==2,0], X[y_hc==2,1],s=100, c = 'green', label = 'Target')
plt.scatter(X[y_hc==3,0], X[y_hc==3,1],s=100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc==4,0], X[y_hc==4,1],s=100, c = 'magenta', label = 'Sensible')


plt.title('Cluster of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
#%%
