import tensorflow.keras.datasets.imdb as imdb
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as reg
import tensorflow.keras.losses as losses
import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np

cant_de_palabras = 10000


def vectorize(sequences, dimension=cant_de_palabras):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1
	return results


def TP4_3a(data, targets,epochs):
	data = vectorize(data)
	targets = np.array(targets).astype("float32")

	test_x = data[:10000]
	test_y = targets[:10000]
	train_x = data[10000:]
	train_y = targets[10000:]

	model = models.Sequential()

	model.add(layers.Dense(50, activation="relu", input_shape=(train_x.shape[1],)))
	# Hidden - Layers
	# model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
	# model.add(layers.Dense(50, activation = "relu"))
	# # model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
	model.add(layers.Dense(25, activation = "relu"))
	# Output- Layer
	model.add(layers.Dense(1, activation = "sigmoid"))
	model.summary()

	optimizer = tf.keras.optimizers.Adadelta(learning_rate=.1)
	lossMSE = losses.MSE
	model.compile(optimizer, loss = lossMSE, metrics = ["accuracy"])
	epocas = epochs
	results = model.fit(train_x, train_y, batch_size=128, epochs=epocas,verbose=1, validation_data=(test_x,test_y))

	plt.figure(1)
	plt.plot(np.arange(epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(epocas), results.history['accuracy'], 'k*-')
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('IMDB: Acc - Sin regularización')
	plt.xlabel('Época')
	plt.ylabel('Accuracy')
	plt.xlim([0, 20])
	plt.ylim([0, 1])
	plt.savefig('TP4_3aAcc.pdf')
	plt.figure(2)
	plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(epocas), results.history['loss'], 'k*-')
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('IMDB: Loss - Sin regularización')
	plt.xlabel('Época')
	plt.ylabel('Loss')
	plt.xlim([0, 20])
	# plt.ylim([10, 50])
	plt.savefig('TP4_3aLoss.pdf')

def TP4_3b(data, targets, epochs):
	data = vectorize(data)
	targets = np.array(targets).astype("float32")

	test_x = data[:10000]
	test_y = targets[:10000]
	train_x = data[10000:]
	train_y = targets[10000:]

	model = models.Sequential()

	model.add(layers.Dense(50, activation="relu", input_shape=(train_x.shape[1],), kernel_regularizer=reg.l2(1e-2)))
	# Hidden - Layers
	# model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
	# model.add(layers.Dense(50, activation = "relu", kernel_regularizer=reg.l2(1e-2)))
	# # model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
	model.add(layers.Dense(25, activation = "relu", kernel_regularizer=reg.l2(1e-2)))
	# Output- Layer
	model.add(layers.Dense(1, activation = "sigmoid", kernel_regularizer=reg.l2(1e-2)))
	model.summary()

	optimizer = tf.keras.optimizers.Adadelta(learning_rate=.1)
	lossMSE = losses.MSE
	model.compile(optimizer, loss=lossMSE, metrics=["accuracy"])
	epocas = epochs
	results = model.fit(train_x, train_y, batch_size=128, epochs=epocas,verbose=1, validation_data=(test_x,test_y))

	plt.figure(3)
	plt.plot(np.arange(epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(epocas), results.history['accuracy'], 'k*-')
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('IMDB: Acc - L2')
	plt.xlabel('Época')
	plt.ylabel('Accuracy')
	plt.xlim([0, 20])
	plt.ylim([0, 1])
	plt.savefig('TP4_3bAcc.pdf')
	plt.figure(4)
	plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(epocas), results.history['loss'], 'k*-')
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('IMDB: Loss - L2')
	plt.xlabel('Época')
	plt.ylabel('Loss')
	plt.xlim([0, 20])
	# plt.ylim([10, 50])
	plt.savefig('TP4_3bLoss.pdf')



def TP4_3c(data, targets, epochs):
	data = vectorize(data)
	targets = np.array(targets).astype("float32")

	test_x = data[:10000]
	test_y = targets[:10000]
	train_x = data[10000:]
	train_y = targets[10000:]

	model = models.Sequential()

	model.add(layers.Dense(50, activation="relu", input_shape=(train_x.shape[1],)))
	# Hidden - Layers
	model.add(layers.BatchNormalization())
	model.add(layers.Dense(25, activation = "relu"))
	model.add(layers.BatchNormalization())
	# model.add(layers.Dense(50, activation = "relu"))
	# model.add(layers.BatchNormalization())
	# Output- Layer
	model.add(layers.Dense(1, activation = "sigmoid"))
	model.summary()

	optimizer = tf.keras.optimizers.Adadelta(learning_rate=.01)
	lossMSE = losses.MSE
	model.compile(optimizer, loss=lossMSE, metrics=["accuracy"])
	epocas = epochs
	results = model.fit(train_x, train_y, batch_size=128, epochs=epocas,verbose=1, validation_data=(test_x,test_y))

	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(epocas), results.history['accuracy'], 'k*-')
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('IMDB: Acc - BN')
	plt.xlabel('Época')
	plt.ylabel('Accuracy')
	plt.xlim([0, 20])
	plt.ylim([0, 1])
	plt.savefig('TP4_3cAcc.pdf')
	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(epocas), results.history['loss'], 'k*-')
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('IMDB: Loss - BN')
	plt.xlabel('Época')
	plt.ylabel('Loss')
	plt.xlim([0, 20])
	# plt.ylim([10, 50])
	plt.savefig('TP4_3cLoss.pdf')


def TP4_3d(data, targets, epochs):
	data = vectorize(data)
	targets = np.array(targets).astype("float32")

	test_x = data[:10000]
	test_y = targets[:10000]
	train_x = data[10000:]
	train_y = targets[10000:]

	model = models.Sequential()

	model.add(layers.Dense(50, activation="relu", input_shape=(train_x.shape[1],)))
	# Hidden - Layers
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(25, activation = "relu"))
	model.add(layers.Dropout(rate=0.5))
	# model.add(layers.Dense(50, activation = "relu"))
	# model.add(layers.BatchNormalization())
	# Output- Layer
	model.add(layers.Dense(1, activation = "sigmoid"))
	model.summary()

	optimizer = tf.keras.optimizers.Adadelta(learning_rate=.1)
	lossMSE = losses.MSE
	model.compile(optimizer, loss=lossMSE, metrics=["accuracy"])
	epocas = epochs
	results = model.fit(train_x, train_y, batch_size=128, epochs=epocas,verbose=1, validation_data=(test_x,test_y))

	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(epocas), results.history['accuracy'], 'k*-')
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('IMDB: Acc - Drop out')
	plt.xlabel('Época')
	plt.ylabel('Accuracy')
	plt.xlim([0, 20])
	plt.ylim([0, 1])
	plt.savefig('TP4_3dAcc.pdf')
	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(epocas), results.history['loss'], 'k*-')
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('IMDB: Loss - Drop out')
	plt.xlabel('Época')
	plt.ylabel('Loss')
	plt.xlim([0, 20])
	# plt.ylim([10, 50])
	plt.savefig('TP4_3dLoss.pdf')


(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)

data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
'''
INDEX_FROM = 0
index = imdb.get_word_index()
reverse_index = dict([(value, key) for (key, value) in index.items()])
decoded = " ".join( [reverse_index.get(i - 3, "#") for i in data[0]])
# print(decoded)
'''
# TP4_3a(data, targets, 20)  # Original
# TP4_3b(data, targets, 20)  # L2
TP4_3c(data, targets, 20)  # BN
# TP4_3d(data, targets, 20)  # Dropout

plt.show()
