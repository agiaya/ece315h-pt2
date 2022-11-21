import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from read_csv import *

data = Database("training_kindashort")
data_array = np.transpose(data.return_array())
data_result = data.return_result()

n_components = 3
tsne = TSNE(n_components)
tsne_result = tsne.fit_transform(data_array)
print(tsne_result.shape)

x = np.matmul(np.transpose(data_array),tsne_result)

'''plt.figure(figsize=(10,10))
plt.scatter(x[:,0],x[:,1],c=data_result,cmap='plasma')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()'''

fig = plt.figure(figsize=(10,10))
 
# choose projection 3d for creating a 3d graph
axis = fig.add_subplot(111, projection='3d')
 
#x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
axis.scatter(x[:,0],x[:,1],x[:,2], c=data_result,cmap='plasma')
axis.set_xlabel("PC1", fontsize=10)
axis.set_ylabel("PC2", fontsize=10)
axis.set_zlabel("PC3", fontsize=10)
plt.show()