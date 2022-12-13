from facial_68_Landmark import *
import numpy as np
from read_csv import Database
from read_csv import Subject
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical



# Initialize Model
model = Sequential([
  Dense(64, activation='relu', input_shape=(68,)),
  Dense(11, activation='softmax'),
])

model.load_weights('model.h5')

