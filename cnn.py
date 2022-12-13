import tensorflow as tf 
from read_csv import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
import os 

'''
data = Database("training_short")

# Model tuning parameters
batch_size = 128
num_classes = 10
epochs = 12
'''
train = Database("training_10000")
train_array = train.return_array()
train_targets = train.return_target()
print(train_array.shape)
print(train_targets.shape)

test = Database("test_short")
test_array = test.return_array()
test_targets = test.return_target()
print(train_array.shape)
print(train_targets.shape)

# Buld the model
model = Sequential([
  Dense(64, activation='relu', input_shape=(68,)),
  Dense(64, activation='relu'),
  Dense(11, activation='softmax'),
])

# Compile the model.
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# Create dictionary of target classes
label_dict = {
 0: '0',
 1: '1',
 2: '2',
 3: '3',
 4: '4',
 5: '5',
 6: '6',
 7: '7',
 8: '8',
 9: '9',
 10: '10',
}

