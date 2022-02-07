# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')

#Looking at the dataset
#Data is received from client card with purchase history
#Spending score is score assigned to clients based on 1) Income 2) Frequency of mall visitation 3) Amount spent at the mall in a year
# Spending score 0(less spent), 100(more spent)
#Aim is to segment(group) clients based on spending score and annual income
#There is no idea of what these segments will be (or how many segments there will be), therefore it is a clustering problem
X = dataset.iloc[:,3:].values

#Choosing optimal number of clusters using the elbow method
from sklearn.cluster import KMeans

#We create a for loop to create 10 different cluster sizes to calculate WCSS
wcss=[]

for i in range (1,11):
# We perform two operations
# 1) Fit K-means to data X
# 2) Compute WCSS and append to wcss list

    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10, random_state=0)
    #k-means++ avoids random initialization

#Fitting the KMeans algorithm to data X
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    #kmeans.inertia_ calculates wcss value
plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Optimal clusters is 5

#Applying KMeans to the dataset with correct number of clusters
kmeans=KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10, random_state=0)
y_kmeans=kmeans.fit_predict(X)

#Fit_predict will tell us the cluster each client belongs

#Visualizing the clusters
#Note for multidimensional problem, this section is not applicable
plt.scatter(X[y_kmeans==0,0], X[y_kmeans==0,1],s=100, c = 'red', label = 'Careful')
plt.scatter(X[y_kmeans==1,0], X[y_kmeans==1,1],s=100, c = 'blue', label = 'Standard')
plt.scatter(X[y_kmeans==2,0], X[y_kmeans==2,1],s=100, c = 'green', label = 'Target')
plt.scatter(X[y_kmeans==3,0], X[y_kmeans==3,1],s=100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_kmeans==4,0], X[y_kmeans==4,1],s=100, c = 'magenta', label = 'Sensible')

#Code breakdown
#y_kmeans is array of clusters
#y_kmeans==a, will return array of booleans where 'a' is true i.e y_kmeans == 0 will give 200X1 array where cluster is '0')
#X[y_kmeans==0,0] all points that are labeled '0' in column 0 of X
#X[y_kmeans==0,1] all points labeled '1' in column 1 of X
#Remember X  is 200 X 2 matrix to plot coordinates, so we are essentially listing two 200 X 1 matrices to list the coordinates of points

#Plotting the cluster centres
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Cluster of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

#Code breakdown
#kmeans.cluster_centers_ returns cluster centre coordinates
#kmeans.cluster_centers_[:,0] returns column 0 of coordinates
#kmeans.cluster_centers_[:,0] returns column 1 of coordinates