import numpy as np
from readCSV import Database
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Nadam
from keras.utils import to_categorical
from keras.losses import kl_divergence

train = Database("training_exp")
train_array = train.return_vector()
train_targets = train.return_target()
print(train_array.shape)
print(train_targets.shape)

test = Database("test_short")
test_array = test.return_vector()
test_targets = test.return_target()
print(train_array.shape)
print(train_targets.shape)

# Build the model.
model = Sequential([
  Dense(64, activation='relu', input_shape=(68,)),
  Dense(11, activation='softmax'),
])

# Compile the model.
model.compile(
  optimizer='adam',
  loss='kl_divergence',
  metrics=['accuracy'],
)

# Train the model.
model.fit(
  train_array,
  to_categorical(train_targets),
  epochs=100,
  batch_size=32,
)

# Evaluate the model.
model.evaluate(
  test_array,
  to_categorical(test_targets)
)

# Save the model to disk.
model.save_weights('model.h5')

# Load the model from disk later using:
# model.load_weights('model.h5')

# Predict on the first 5 test images.
predictions = model.predict(test_array[:100])

# Print our model's predictions.
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print(test_targets[:100]) # [7, 2, 1, 0, 4]