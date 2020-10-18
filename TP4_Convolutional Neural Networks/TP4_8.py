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


def arqDensa(x_train, y_train, x_test, y_test):
	n_train_data = x_train.shape[0]  # Cantidad de datos que se usarán para entrenar
	x_test = x_test.astype("float32")
	x_train = x_train.astype("float32")
	# acondiciono los datos de training
	im_shape = x_train.shape[1:]
	xtr = np.reshape(x_train[:n_train_data],
	                 (x_train[:n_train_data].shape[0], np.prod(im_shape)))  # Los datos de train como vector

	x_mean = np.mean(xtr, axis=0)[np.newaxis, :]  # media de los datos de training
	std = np.std(xtr, axis=0)[np.newaxis, :]  # STD de los datos de training

	xtr = (xtr) / 255

	ytr = y_train[:n_train_data]
	yy_tr = np.zeros((ytr.shape[0], 10))  # esto es un vector de train_data x 10 para representar las clases
	yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # vector de train_data x 10

	# Acondiciono los datos de testing:
	n_train_data_t = int(n_train_data / 10)  # los datos que se van a tomar para el testing
	im_shape_test = x_test.shape[1:]
	xt = np.reshape(x_test[:n_train_data_t],
	                (x_test[:n_train_data_t].shape[0], np.prod(im_shape_test)))  # Los datos de test como vector

	xt = (xt) / 255
	yt = y_test[:n_train_data_t]
	yy_t = np.zeros((yt.shape[0], 10))  # esto es un vector de train_data x 10 para representar las clases
	yy_t[np.arange(yt.shape[0]), yt.T] = 1  # vector de train_data x 10

	model = models.Model()
	input = layers.Input(shape=[28*28])
	d = layers.Dense(100, activation=activations.relu)(input)
	# d = layers.Dropout(0.15)(d)
	# d = layers.Dense(100, activation=activations.relu)(d)
	d = layers.Dropout(0.15)(d)
	out = layers.Dense(10,  activation=activations.sigmoid)(d)
	model = tf.keras.models.Model(input, out)

	model.summary()

	optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)
	model.compile(optimizer, loss=losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
	epocas = 20
	results = model.fit(xtr, yy_tr, batch_size=128, epochs=epocas, verbose=2, validation_data=(xt, yy_t))

	plt.figure(3)
	plt.plot(np.arange(epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(epocas), results.history['accuracy'], 'k*-')
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('Ejercicio 8: Arq. de capas densas')
	plt.xlabel('Época')
	plt.ylim([0, 1])
	plt.ylabel('Accuracy')
	plt.savefig('TP4_8_D_acc.pdf')

	plt.figure(4)
	plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(epocas), results.history['loss'], 'k*-')
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('Ejercicio 8: Arq. de capas Densas')
	plt.xlabel('Época')
	plt.ylabel('Loss')
	plt.savefig('TP4_8_D_Loss.pdf')
	plt.plot()


def arqConv(x_train, y_train, x_test, y_test):
	n_train_data = x_train.shape[0]  # Cantidad de datos que se usarán para entrenar
	x_test = x_test.astype("float32")
	x_train = x_train.astype("float32")
	# acondiciono los datos de training
	im_shape = x_train.shape[1:]
	xtr = np.reshape(x_train[:n_train_data],
	                 (x_train[:n_train_data].shape[0], np.prod(im_shape)))  # Los datos de train como vector

	x_mean = np.mean(xtr, axis=0)[np.newaxis, :]  # media de los datos de training
	std = np.std(xtr, axis=0)[np.newaxis, :]  # STD de los datos de training

	xtr = (xtr) / 255

	ytr = y_train[:n_train_data]
	yy_tr = np.zeros((ytr.shape[0], 10))  # esto es un vector de train_data x 10 para representar las clases
	yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # vector de train_data x 10

	# Acondiciono los datos de testing:
	n_train_data_t = int(n_train_data / 10)  # los datos que se van a tomar para el testing
	im_shape_test = x_test.shape[1:]
	xt = np.reshape(x_test[:n_train_data_t],
	                (x_test[:n_train_data_t].shape[0], np.prod(im_shape_test)))  # Los datos de test como vector

	xt = (xt) / 255
	yt = y_test[:n_train_data_t]
	yy_t = np.zeros((yt.shape[0], 10))  # esto es un vector de train_data x 10 para representar las clases
	yy_t[np.arange(yt.shape[0]), yt.T] = 1  # vector de train_data x 10

	xtr = np.reshape(xtr, (-1, 28, 28, 1))
	xt = np.reshape(xt, (-1, 28, 28, 1))

	models.Model()
	input_img = layers.Input(shape=xtr.shape[1:])
	x = layers.Conv2D(16, (3, 3), activation='relu')(input_img)
	x = layers.Dropout(.5)(x)
	x = layers.Conv2D(16, (3, 3), activation='relu')(x)

	x = layers.Flatten()(x)
	x = layers.Dropout(.5)(x)
	out = layers.Dense(10, activation='sigmoid')(x)

	autoencoder = models.Model(input_img, out)

	autoencoder.summary()

	optimizer = tf.keras.optimizers.Adagrad(learning_rate=.01)
	autoencoder.compile(optimizer, loss=losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
	epocas = 20
	results = autoencoder.fit(xtr, yy_tr, batch_size=64, epochs=epocas, verbose=2, validation_data=(xt, yy_t))

	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(epocas), results.history['accuracy'], 'k*-')
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('Ejercicio 8: Arq. de capas convolucionales')
	plt.xlabel('Época')
	plt.ylim([0, 1])
	plt.ylabel('Accuracy')
	plt.savefig('TP4_8_C_acc.pdf')

	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(epocas), results.history['loss'], 'k*-')
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('Ejercicio 8: Arq. de capas convolucionales')
	plt.xlabel('Época')
	plt.ylabel('Loss')
	plt.savefig('TP4_8_C_Loss.pdf')
	plt.plot()



(x_train, y_train), (x_test, y_test) = mnist.load_data()  # Cargo los datos de CIFAR-10
#arqDensa(x_train, y_train, x_test, y_test)
arqConv(x_train, y_train, x_test, y_test)
plt.show()