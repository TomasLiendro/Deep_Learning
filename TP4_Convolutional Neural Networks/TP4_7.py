import tensorflow.keras.datasets.mnist as mnist
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.regularizers as reg
import tensorflow.keras.losses as losses
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Cargo los datos de CIFAR-10
n_train_data = 20000     # Cantidad de datos que se usarán para entrenar

# acondiciono los datos de training
im_shape = x_train.shape[1:]
xtr = np.reshape(x_train[:n_train_data], (x_train[:n_train_data].shape[0], np.prod(im_shape)))   # Los datos de train como vector

x_mean = np.mean(xtr, axis=0)[np.newaxis, :]  # media de los datos de training
std = np.std(xtr, axis=0)[np.newaxis, :]    # STD de los datos de training

xtr = xtr / 255
xtr_n = xtr + np.random.normal(0, 0.5, size=(n_train_data, np.prod(im_shape))) * 0.5
xtr_n = np.clip(xtr_n, 0, 1)

ytr = y_train[:n_train_data]

# Acondiciono los datos de testing:
n_train_data_t = int(n_train_data/10)  # los datos que se van a tomar para el testing
im_shape_test = x_test.shape[1:]
xt = np.reshape(x_test[:n_train_data_t], (x_test[:n_train_data_t].shape[0], np.prod(im_shape_test)))  # Los datos de test como vector

xt = xt / 255
xt_n = xt + np.random.normal(0, 0.5, size=(n_train_data_t, np.prod(im_shape_test))) * 0.5
xt_n = np.clip(xt_n, 0, 1)
yt = y_test[:n_train_data_t]


model = models.Model()
input = layers.Input(shape=[28*28])
d = layers.Dense(1000, activation=activations.relu)(input)
d = layers.Dense(500, activation=activations.relu)(d)
d = layers.Dense(250, activation=activations.relu)(d)
d = layers.Dense(500, activation=activations.relu)(d)
d = layers.Dense(1000, activation=activations.relu)(d)
out = layers.Dense(28*28, activation=activations.sigmoid)(d)
model = tf.keras.models.Model(input, out)


model.summary()

optimizer = tf.keras.optimizers.Adadelta(learning_rate=1)
model.compile(optimizer, loss='binary_crossentropy', metrics=['accuracy'])
epocas = 20
results = model.fit(xtr_n, xtr, batch_size=128, epochs=epocas, verbose=1, validation_data=(xt_n, xt))

a = model.predict(xt_n)
n=5
plt.figure(figsize=(n, 3))
for i in range(n):
    # Original
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test[i+30].reshape(28, 28))
    plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # Con ruido
    ax = plt.subplot(3, n, i+1+n)
    plt.imshow(xt_n[i+30].reshape(28, 28))
    plt.gray()
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # Filtrado
    ax = plt.subplot(3, n, i+1+2*n)
    plt.imshow(a[i+30].reshape(28, 28))
    plt.gray()

plt.savefig('TP4_7_imagenes.pdf')
plt.show()
