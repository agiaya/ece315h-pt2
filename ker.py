import keras 

from keras.datasets import boston_housing 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.optimizers import RMSprop 
from keras.callbacks import EarlyStopping 
from sklearn import preprocessing 
from sklearn.preprocessing import scale

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
x_train_scaled = preprocessing.scale(x_train) 
scaler = preprocessing.StandardScaler().fit(x_train) 
x_test_scaled = scaler.transform(x_test)

model = Sequential() 
model.add(Dense(64, kernel_initializer = 'normal', activation = 'relu', input_shape = (13,))) 
model.add(Dense(64, activation = 'relu')) 
model.add(Dense(1))

model.compile(
   loss = 'mse', 
   optimizer = RMSprop(), 
   metrics = ['mean_absolute_error']
)

history = model.fit(
   x_train_scaled, y_train,    
   batch_size=128, 
   epochs = 500, 
   verbose = 1, 
   validation_split = 0.2, 
   callbacks = [EarlyStopping(monitor = 'val_loss', patience = 20)]
)

score = model.evaluate(x_test_scaled, y_test, verbose = 0) 
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

prediction = model.predict(x_test_scaled) 
print(prediction.flatten()) 
print(y_test)