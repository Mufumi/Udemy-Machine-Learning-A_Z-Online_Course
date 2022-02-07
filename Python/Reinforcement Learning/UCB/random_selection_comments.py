# Random Selection

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Random Selection
import random
N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
# Create a variable that chooses a random advert
    ad = random.randrange(d)
# Add advert into list
    ads_selected.append(ad)
# Check if the advert listed has a reward (will add if there is a reward)
    reward = dataset.values[n, ad]
# Addition of all rewards
    total_reward = total_reward + reward

# Visualising the results
# plt.hist plots all the listed adverts
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()