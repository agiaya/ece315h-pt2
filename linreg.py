import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from readCSV import Database

data = Database("training")
data_array = data.return_array()
data_targets = data.return_target()
print(data_array.shape)
print(data_targets.shape)

reg = LinearRegression().fit(data_array,data_targets)
print(reg.score(data_array,data_targets))
