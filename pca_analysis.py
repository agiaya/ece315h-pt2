# running PCA Analysis on the dimensions from the data set
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from read_csv import Database

data = Database("training_short").return_array()
