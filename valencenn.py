import numpy as np
from readCSV import Database
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Nadam
from keras.utils import normalize

train = Database("training_exp")
train_array = train.return_vector()
train_targets = train.return_valence()
print(train_array.shape)
print(train_targets.shape)

test = Database("test_short")
test_array = test.return_vector()
test_targets = test.return_valence()
print(train_array.shape)
print(train_targets.shape)

# Build the model.
model = Sequential([
  Dense(64, activation='relu', input_shape=(68,)),
  Dense(16, activation='relu'),
  Dense(8, activation='relu'),
  Dense(1, activation='tanh'),
])

# Compile the model.
model.compile(
  optimizer='nadam',
  loss='mse',
  metrics=['mean_absolute_error'],
)

# Train the model.
model.fit(
  train_array,
  train_targets,
  epochs=10,
  batch_size=32,
)

# Evaluate the model.
model.evaluate(
  test_array,
  test_targets
)

# Save the model to disk.
model.save_weights('valencemodel.h5')

# Load the model from disk later using:
# model.load_weights('model.h5')

# Predict on the first 5 test images.
predictions = model.predict(test_array[:20])

# Print our model's predictions.
list = []
for item in predictions:
    list.append(item[0])
print(np.array(list))

# Check our predictions against the ground truths.
print(test_targets[:20]) # [7, 2, 1, 0, 4]