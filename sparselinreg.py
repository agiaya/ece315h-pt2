import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from read_csv import Database

data = Database("training_kindashort")
data_array = data.return_array()
data_targets = data.return_results()
print(data_array.shape)
print(data_targets.shape)

reg = linear_model.Lasso(alpha=0.1)
reg = reg.fit(data_array,data_targets)
print(reg.score(data_array,data_targets))