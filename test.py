import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
#from read_csv import Database
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
data.keys()

df1=pd.DataFrame(data['data'],columns=data['feature_names'])
 
# Scale data before applying PCA
scaling=StandardScaler()
 
# Use fit and transform method
scaling.fit(df1)
Scaled_data=scaling.transform(df1)

print(data)