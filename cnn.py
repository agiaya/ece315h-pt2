from keras.datasets import keras
import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from read_csv import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D

data = Database("training_short")

# Model tuning parameters
batch_size = 128
num_classes = 10
epochs = 12


