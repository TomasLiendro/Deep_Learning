import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.datasets.cifar10 as cifar10

import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as reg
import tensorflow.keras.regularizers as regression
import tensorflow.keras.optimizers as opt
import tensorflow.keras.losses as losses
import tensorflow.keras.activations as activations
import tensorflow.keras.initializers as initializers

import tensorflow as tf


def xor_acc(y_true=None, y_pred=None):
	acc = tf.reduce_mean(tf.cast(tf.less_equal(tf.abs(y_true - y_pred), 0.1), tf.float32))
	return acc


def TP2_3(x_train, y_train, x_test, y_test):
	n_train_data = 20000  # Cantidad de datos que se usarán para entrenar
	nclases = 10  # salida
	nintermedia = 100  # intermedia
	batch_size = 200  # Batch size
	n_epocas = 100  # Cantidad de épocas
	learning_rate = 1e-3  # Learning rate
	reg_lambda = 1e-3  # Coeficiente de regularización

	# acondiciono los datos de training
	im_shape = x_train.shape[1:]
	xtr = np.reshape(x_train[:n_train_data],
	                 (x_train[:n_train_data].shape[0], np.prod(im_shape)))  # Los datos de train como vector

	x_mean = np.mean(xtr, axis=0)[np.newaxis, :]  # media de los datos de training
	std = np.std(xtr, axis=0)[np.newaxis, :]  # STD de los datos de training

	xtr = (xtr - x_mean) / std
	ytr = y_train[:n_train_data]
	yy_tr = np.zeros((ytr.shape[0], nclases))  # esto es un vector de train_data x 10 para representar las clases
	yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # vector de train_data x 10

	# Acondiciono los datos de testing:
	n_train_data_t = int(n_train_data / 10)  # los datos que se van a tomar para el testing
	im_shape_test = x_test.shape[1:]
	xt = np.reshape(x_test[:n_train_data_t],
	                (x_test[:n_train_data_t].shape[0], np.prod(im_shape_test)))  # Los datos de test como vector
	xt = (xt - x_mean) / std
	yt = y_test[:n_train_data_t]

	yy_t = np.zeros((yt.shape[0], nclases))  # esto es un vector de train_data x 10 para representar las clases
	yy_t[np.arange(yt.shape[0]), yt.T] = 1  # vector de train_data x 10

	model = models.Sequential()

	# Primera capa:
	model.add(layers.Dense(units=nintermedia, input_shape=(xtr.shape[1],), activation=activations.sigmoid,
	                       kernel_regularizer=reg.l2(reg_lambda), use_bias=True))
	# capa:
	model.add(layers.Dense(units=nclases, activation=activations.linear, kernel_regularizer=reg.l2(reg_lambda),
	                       use_bias=True))
	model.summary()

	optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
	model.compile(optimizer, loss="mse", metrics=["accuracy"])
	results = model.fit(xtr, yy_tr, batch_size=batch_size, epochs=n_epocas, verbose=1, validation_data=(xt, yy_t))

	plt.figure()
	plt.plot(np.arange(n_epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(n_epocas), results.history['accuracy'], 'k*-')
	plt.figure()
	plt.plot(np.arange(n_epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(n_epocas), results.history['loss'], 'k*-')


def TP2_3b(x_train, y_train, x_test, y_test):
	n_train_data = 20000  # Cantidad de datos que se usarán para entrenar
	nclases = 10  # salida
	nintermedia = 100  # intermedia
	batch_size = 200  # Batch size
	n_epocas = 50  # Cantidad de épocas
	learning_rate = 1e-2  # Learning rate
	reg_lambda = 1e-3  # Coeficiente de regularización

	# acondiciono los datos de training
	im_shape = x_train.shape[1:]
	xtr = np.reshape(x_train[:n_train_data],
	                 (x_train[:n_train_data].shape[0], np.prod(im_shape)))  # Los datos de train como vector

	x_mean = np.mean(xtr, axis=0)[np.newaxis, :]  # media de los datos de training
	std = np.std(xtr, axis=0)[np.newaxis, :]  # STD de los datos de training

	xtr = (xtr - x_mean) / std
	ytr = y_train[:n_train_data]
	yy_tr = np.zeros((ytr.shape[0], nclases))  # Esto es un vector de train_data x 10 para representar las clases
	yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # Vector de train_data x 10

	# Acondiciono los datos de testing:
	n_train_data_t = int(n_train_data / 10)  # los datos que se van a tomar para el testing
	im_shape_test = x_test.shape[1:]
	xt = np.reshape(x_test[:n_train_data_t],
	                (x_test[:n_train_data_t].shape[0], np.prod(im_shape_test)))  # Los datos de test como vector
	xt = (xt - x_mean) / std
	yt = y_test[:n_train_data_t]

	yy_t = np.zeros((yt.shape[0], nclases))  # esto es un vector de train_data x 10 para representar las clases
	yy_t[np.arange(yt.shape[0]), yt.T] = 1  # vector de train_data x 10

	model = models.Sequential()

	# Primera capa:
	model.add(layers.Dense(units=nintermedia, input_shape=(xtr.shape[1],), activation=activations.sigmoid,
	                       kernel_regularizer=reg.l2(reg_lambda), use_bias=True))
	# capa:
	model.add(layers.Dense(units=nclases, activation=activations.linear, kernel_regularizer=reg.l2(reg_lambda),
	                       use_bias=True))
	model.summary()

	optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
	model.compile(optimizer, loss=losses.hinge, metrics=["accuracy"])
	results = model.fit(xtr, yy_tr, batch_size=batch_size, epochs=n_epocas, verbose=1, validation_data=(xt, yy_t))

	plt.figure()
	plt.plot(np.arange(n_epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(n_epocas), results.history['accuracy'], 'k*-')
	plt.figure()
	plt.plot(np.arange(n_epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(n_epocas), results.history['loss'], 'k*-')


def TP2_3c(x_train, y_train, x_test, y_test):
	n_train_data = 20000  # Cantidad de datos que se usarán para entrenar
	nclases = 10  # salida
	nintermedia = 100  # intermedia
	batch_size = 50  # Batch size
	n_epocas = 50  # Cantidad de épocas
	learning_rate = 1e-2  # Learning rate
	reg_lambda = 1e-4  # Coeficiente de regularización

	# acondiciono los datos de training
	im_shape = x_train.shape[1:]
	xtr = np.reshape(x_train[:n_train_data],
	                 (x_train[:n_train_data].shape[0], np.prod(im_shape)))  # Los datos de train como vector

	x_mean = np.mean(xtr, axis=0)[np.newaxis, :]  # media de los datos de training
	std = np.std(xtr, axis=0)[np.newaxis, :]  # STD de los datos de training

	xtr = (xtr - x_mean) / std
	ytr = y_train[:n_train_data]
	yy_tr = np.zeros((ytr.shape[0], nclases))  # Esto es un vector de train_data x 10 para representar las clases
	yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # Vector de train_data x 10

	# Acondiciono los datos de testing:
	n_train_data_t = int(n_train_data / 10)  # los datos que se van a tomar para el testing
	im_shape_test = x_test.shape[1:]
	xt = np.reshape(x_test[:n_train_data_t],
	                (x_test[:n_train_data_t].shape[0], np.prod(im_shape_test)))  # Los datos de test como vector
	xt = (xt - x_mean) / std
	yt = y_test[:n_train_data_t]

	yy_t = np.zeros((yt.shape[0], nclases))  # esto es un vector de train_data x 10 para representar las clases
	yy_t[np.arange(yt.shape[0]), yt.T] = 1  # vector de train_data x 10

	model = models.Sequential()

	# Primera capa:
	model.add(layers.Dense(units=nintermedia, input_shape=(xtr.shape[1],), activation=activations.sigmoid,
	                       kernel_regularizer=reg.l2(reg_lambda), use_bias=True, kernel_initializer=initializers.RandomNormal(stddev=3)))
	# capa:
	model.add(layers.Dense(units=nclases, activation=activations.linear, kernel_regularizer=reg.l2(reg_lambda),
	                       use_bias=True, kernel_initializer=initializers.RandomNormal(stddev=1)))
	model.summary()

	optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
	model.compile(optimizer, loss=losses.categorical_crossentropy, metrics=["accuracy"])
	results = model.fit(xtr, yy_tr, batch_size=batch_size, epochs=n_epocas, verbose=1, validation_data=(xt, yy_t))

	plt.figure()
	plt.plot(np.arange(n_epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(n_epocas), results.history['accuracy'], 'k*-')
	plt.figure()
	plt.plot(np.arange(n_epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(n_epocas), results.history['loss'], 'k*-')


def TP2_4(x_train, y_train, x_test, y_test):
	n_train_data = 20000  # Cantidad de datos que se usarán para entrenar
	nclases = 10  # salida
	nintermedia = 100  # intermedia
	batch_size = 50  # Batch size
	n_epocas = 50  # Cantidad de épocas
	learning_rate = 1e-2  # Learning rate
	reg_lambda = 1e-4  # Coeficiente de regularización

	# acondiciono los datos de training
	im_shape = x_train.shape[1:]
	xtr = np.reshape(x_train[:n_train_data],
	                 (x_train[:n_train_data].shape[0], np.prod(im_shape)))  # Los datos de train como vector

	x_mean = np.mean(xtr, axis=0)[np.newaxis, :]  # media de los datos de training
	std = np.std(xtr, axis=0)[np.newaxis, :]  # STD de los datos de training

	xtr = (xtr - x_mean) / std
	ytr = y_train[:n_train_data]
	yy_tr = np.zeros((ytr.shape[0], nclases))  # Esto es un vector de train_data x 10 para representar las clases
	yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # Vector de train_data x 10

	# Acondiciono los datos de testing:
	n_train_data_t = int(n_train_data / 10)  # los datos que se van a tomar para el testing
	im_shape_test = x_test.shape[1:]
	xt = np.reshape(x_test[:n_train_data_t],
	                (x_test[:n_train_data_t].shape[0], np.prod(im_shape_test)))  # Los datos de test como vector
	xt = (xt - x_mean) / std
	yt = y_test[:n_train_data_t]

	yy_t = np.zeros((yt.shape[0], nclases))  # esto es un vector de train_data x 10 para representar las clases
	yy_t[np.arange(yt.shape[0]), yt.T] = 1  # vector de train_data x 10

	model = models.Sequential()

	# Primera capa:
	model.add(layers.Dense(units=nintermedia, input_shape=(xtr.shape[1],), activation=activations.sigmoid,
	                       kernel_regularizer=reg.l2(reg_lambda), use_bias=True, kernel_initializer=initializers.RandomNormal(stddev=3)))
	# capa:
	model.add(layers.Dense(units=nclases, activation=activations.linear, kernel_regularizer=reg.l2(reg_lambda),
	                       use_bias=True, kernel_initializer=initializers.RandomNormal(stddev=1)))
	model.summary()

	optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
	model.compile(optimizer, loss=losses.categorical_crossentropy, metrics=["accuracy"])
	results = model.fit(xtr, yy_tr, batch_size=batch_size, epochs=n_epocas, verbose=1, validation_data=(xt, yy_t))

	plt.figure()
	plt.plot(np.arange(n_epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(n_epocas), results.history['accuracy'], 'k*-')
	plt.figure()
	plt.plot(np.arange(n_epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(n_epocas), results.history['loss'], 'k*-')


def TP2_6a(x_train, y_train, x_test=None, y_test=None):
	n_train_data = x_train.shape[0]  # Cantidad de datos que se usarán para entrenar
	nclases = 1  # salida
	nintermedia = 2  # intermedia
	batch_size = n_train_data  # Batch size
	n_epocas = 200  # Cantidad de épocas
	learning_rate = 1e-0  # Learning rate
	reg_lambda = 0  # Coeficiente de regularización

	# acondiciono los datos de training
	im_shape = x_train.shape[1:]
	xtr = np.reshape(x_train[:n_train_data],
	                 (x_train[:n_train_data].shape[0], np.prod(im_shape)))  # Los datos de train como vector

	ytr = y_train[:n_train_data]

	# Acondiciono los datos de testing:
	model = models.Sequential()

	# Primera capa:
	x = layers.Input(shape=(2,))
	l1 = model.add(layers.Dense(units=nintermedia, input_shape=(xtr.shape[1],), activation=activations.tanh,
	                       kernel_regularizer=reg.l2(reg_lambda), use_bias=True))
	# capa:
	l2 = model.add(layers.Dense(units=nclases, activation=activations.tanh,use_bias=True))
	# modelo = tf.keras.Model(inputs=x, outputs=l2)
	# model.summary()

	optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
	model.compile(optimizer, loss=losses.MSE, metrics=[xor_acc, 'acc', 'mse'])
	results = model.fit(xtr, ytr, batch_size=batch_size, epochs=n_epocas, verbose=1, validation_data=(xtr, ytr))

	plt.figure()
	# plt.plot(np.arange(n_epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(n_epocas), results.history['xor_acc'], '*-')
	plt.plot(np.arange(n_epocas), results.history['acc'], '*-')
	# plt.plot(np.arange(n_epocas), results.history['mse'], '*-')
	plt.figure()
	# plt.plot(np.arange(n_epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(n_epocas), results.history['loss'], 'k*-')

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()  # Cargo los datos de CIFAR-10
# TP2_3(x_train,y_train,x_test, y_test)
# TP2_3b(x_train, y_train, x_test, y_test)
# TP2_3c(x_train, y_train, x_test, y_test)  # Este no da todavía
# TP2_4(x_train, y_train, x_test, y_test)  # Este no da todavía

x_train = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y_train = np.array([[1], [-1], [-1], [1]])

TP2_6a(x_train, y_train)
plt.show()
