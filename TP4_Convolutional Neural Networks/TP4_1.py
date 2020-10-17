from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizer
import tensorflow.keras.losses as losses
import tensorflow.keras.activations as activations
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np
data, targets = load_boston(return_X_y=True)

test_x = data[:int(data.shape[0]*0.25)]
test_y = targets[:int(data.shape[0]*0.25)]
train_x = data[int(data.shape[0]*0.25):]
train_y = targets[int(data.shape[0]*0.25):]

x_mean = np.mean(train_x, axis=0)
x_std = np.std(train_x, axis=0)
y_norm = np.max(train_y)

test_x = (test_x-x_mean)/x_std
train_x = (train_x-x_mean)/x_std
# '''
# Primera arquitectura
model = models.Sequential()

model.add(layers.Dense(2, activation=activations.linear, input_shape=(train_x.shape[1],)))
model.add(layers.Dense(1, activation=activations.linear, input_shape=(train_x.shape[1],)))

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer, loss=losses.mse, metrics=["accuracy"])
epocas = 50
results = model.fit(train_x, train_y, batch_size=2, epochs=epocas, verbose=1, validation_data=(test_x, test_y))
y_pred = model.predict(test_x)

regression = LinearRegression().fit(test_y[:,np.newaxis], y_pred)

plt.figure()
plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
plt.plot(np.arange(epocas), results.history['loss'], 'k*-')
plt.legend(['validation_data', 'training data'])
plt.title('Inmuebles de Boston - Loss')
plt.xlabel('Época')
plt.ylabel('Función de costo')
plt.savefig('TP4_1aloss.pdf')
plt.figure()
plt.scatter(test_y, y_pred)
x = np.array([np.min(test_y), np.max(test_y)])
y = np.array([x[0] * regression.coef_[0] + regression.intercept_, x[1] * regression.coef_[0] + regression.intercept_])
plt.plot(x, y, 'r', linewidth=3)
plt.legend([ 'Ajuste lineal', 'Valores predichos'])
plt.title('Inmuebles de Boston')
plt.xlabel('Precio real')
plt.ylabel('Precio estimado')
plt.xlim([10, 50])
plt.ylim([10, 50])
print(regression.score(test_y[:,np.newaxis], y_pred))
plt.savefig('TP4_1a.pdf')
plt.show()
'''
# Segunda arquitectura

model = models.Sequential()

model.add(layers.Dense(2, activation=activations.relu, input_shape=(train_x.shape[1],), kernel_regularizer=regularizer.l2(0.0001)))
model.add(layers.Dense(2, activation=activations.relu, kernel_regularizer=regularizer.l2(0.0001)))
model.add(layers.Dense(2, activation=activations.relu, kernel_regularizer=regularizer.l2(0.0001)))
model.add(layers.Dense(2, activation=activations.relu, kernel_regularizer=regularizer.l2(0.0001)))
model.add(layers.Dense(1, activation=activations.relu, kernel_regularizer=regularizer.l2(0.0001)))

model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer, loss=losses.mse, metrics=["accuracy"])
epocas = 50
results = model.fit(train_x, train_y, batch_size=2, epochs=epocas, verbose=1, validation_data=(test_x, test_y))
y_pred = model.predict(test_x)

regression = LinearRegression().fit(test_y[:,np.newaxis], y_pred)

plt.figure()
plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
plt.plot(np.arange(epocas), results.history['loss'], 'k*-')
plt.legend(['validation_data', 'training data'])
plt.figure()
plt.scatter(test_y, y_pred)
x = np.array([np.min(test_y), np.max(test_y)])
y = np.array([x[0] * regression.coef_[0] + regression.intercept_, x[1] * regression.coef_[0] + regression.intercept_])
plt.plot(x, y, 'r', linewidth=3)
plt.legend([ 'Ajuste lineal', 'Valores predichos'])
plt.title('Inmuebles de Boston')
plt.xlabel('Precio real')
plt.ylabel('Precio estimado')
plt.xlim([10, 50])
plt.ylim([10, 50])
print(regression.score(test_y[:,np.newaxis], y_pred))
plt.savefig('TP4_1b.pdf')
plt.show()
# '''