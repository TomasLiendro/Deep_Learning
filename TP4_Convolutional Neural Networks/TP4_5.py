import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as regularizer
import tensorflow.keras.losses as losses
import tensorflow.keras.activations as activations
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np


nro_datos=1000
xi=np.random.uniform(size=1)  # x inicial
datos=[xi[0]]  # evolución en el tiempo
for i in range(nro_datos):
    datos.append(4*datos[-1]*(1-datos[-1]))

datos=np.asarray(datos)
x=datos[:-1]
y=datos[1:]

plt.scatter(x,y)
plt.title('Mapeo logístico: Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('TP4_5_dataset')
# plt.show()
np.random.shuffle(x)
np.random.shuffle(y)
train_x = x[:int(x[:, np.newaxis].shape[0]*9/10)]
train_y = y[:int(y[:, np.newaxis].shape[0]*9/10)]

test_x = x[int(x[:, np.newaxis].shape[0]*9/10):]
test_y = y[int(y[:, np.newaxis].shape[0]*9/10):]

model = models.Model()
input = layers.Input(shape=[1])
c1 = layers.Dense(5, activation=activations.relu, kernel_regularizer=regularizer.l2(0.01))(input)
concatenate = layers.concatenate([c1, input])
out = layers.Dense(1, activation=activations.linear, kernel_regularizer=regularizer.l2(0.01))(concatenate)
model=tf.keras.models.Model(input, out)
model.summary()

optimizer = tf.keras.optimizers.Adadelta(learning_rate=.1)
model.compile(optimizer, loss='mse', metrics=['mse'])
epocas = 100
results = model.fit(train_x, train_y, batch_size=16, epochs=epocas, verbose=1, validation_data=(test_x, test_y))

plt.figure(3)
plt.plot(np.arange(epocas), results.history['val_mse'], 'r*-')
plt.plot(np.arange(epocas), results.history['mse'], 'k*-')
plt.legend(['Datos de validación', 'Datos de entrenamiento'])
plt.title('Mapeo logístico: Métrica')
plt.xlabel('Época')
plt.ylabel('MSE')
plt.xlim([0, 100])
# plt.ylim([10, 50])
plt.savefig('TP4_5metric.pdf')
plt.figure(4)
plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
plt.plot(np.arange(epocas), results.history['loss'], 'k*-')
plt.legend(['Datos de validación', 'Datos de entrenamiento'])
plt.title('Mapeo logístico: Loss')
plt.xlabel('Época')
plt.ylabel('MSE')
plt.xlim([0, 100])
# plt.ylim([10, 50])
plt.savefig('TP4_5Loss.pdf')


plt.show()