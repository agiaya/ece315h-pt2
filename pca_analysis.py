# running PCA Analysis on the dimensions from the data set
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from read_csv import Database

data = Database("training_short").return_array()

scaling = StandardScaler()

scaling.fit(data)
Scaled_data = scaling.transform(data)

principal = PCA(n_components = 3)
principal.fit(Scaled_data)
x = principal.transform(Scaled_data)

print(x)
#print(x.shape)

plt.figure(figsize=(10,10))
plt.scatter(x[:,0],x[:,1],c=data.return_result(),cmap='plasma')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()