# running PCA Analysis on the dimensions from the data set
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from read_csv import Database

data = Database("training_short")
data_array = data.return_array()
data_result = data.return_result()

scaling = StandardScaler()

scaling.fit(data_array)
Scaled_data = scaling.transform(data_array)

principal = PCA(n_components = 2)
principal.fit(Scaled_data)
x = principal.transform(Scaled_data)

#print(x)
print(x.shape)
#np.savetxt("x.csv", x, delimiter=" ")
#print(x.shape)

plt.figure(figsize=(10,10))
plt.scatter(x[:,0],x[:,1],c=data_result,cmap='tab20')
plt.xlabel('pc1')
plt.ylabel('pc2')
#plt.show()
#fig = plt.figure(figsize=(10,10))
 
# choose projection 3d for creating a 3d graph
#axis = fig.add_subplot(111, projection='3d')
 
# x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
#axis.scatter(x[:,0],x[:,1],x[:,2], c=data_result,cmap='plasma')
#axis.set_xlabel("PC1", fontsize=10)
#axis.set_ylabel("PC2", fontsize=10)
#axis.set_zlabel("PC3", fontsize=10)
#plt.show()