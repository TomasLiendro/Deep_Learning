import tensorflow.keras.datasets.imdb as imdb
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.regularizers as reg
import tensorflow.keras.losses as losses
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

cant_de_palabras = 10000


def vectorize(sequences, dimension=cant_de_palabras):
	results = np.zeros((len(sequences), dimension))
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1
	return results


def TP4_4a(data, targets, epochs):
	# data = vectorize(data)
	targets = np.array(targets).astype("float32")

	test_x = data[:10000]
	test_y = targets[:10000]
	train_x = data[10000:]
	train_y = targets[10000:]
	max_seq_l = 100
	test_x = pad_sequences(test_x, maxlen=max_seq_l, value=0.0)
	train_x = pad_sequences(train_x, maxlen=max_seq_l, value=0.0)

	model = models.Sequential()

	model.add(tf.keras.layers.Embedding(cant_de_palabras, 15, input_length=max_seq_l))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(50, activation="relu"))  # , input_shape=(train_x.shape[1],)))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(25, activation="relu"))  # , input_shape=(train_x.shape[1],)))
	model.add(layers.Flatten())
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(1, activation = "sigmoid"))
	model.summary()

	optimizer = tf.keras.optimizers.Adadelta(learning_rate=.1)
	model.compile(optimizer, loss='categorical_cross_entropy', metrics=['accuracy'])
	epocas = epochs
	results = model.fit(train_x, train_y, batch_size=128, epochs=epocas, verbose=1, validation_data=(test_x, test_y))

	plt.figure(3)
	plt.plot(np.arange(epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(epocas), results.history['accuracy'], 'k*-')
	plt.figure(4)
	plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(epocas), results.history['loss'], 'k*-')


def TP4_4b(data, targets, epochs):
	# data = vectorize(data)
	targets = np.array(targets).astype("float32")

	test_x = data[:10000]
	test_y = targets[:10000]
	train_x = data[10000:]
	train_y = targets[10000:]
	max_seq_l = 100
	test_x = pad_sequences(test_x, maxlen=max_seq_l, value=0.0)
	train_x = pad_sequences(train_x, maxlen=max_seq_l, value=0.0)

	model = models.Sequential()

	model.add(layers.Embedding(cant_de_palabras, 15, input_length=max_seq_l))
	model.add(layers.Dropout(0.5))
	# Hidden - Layers
	model.add(layers.Conv1D(filters=4, kernel_size=2, padding="same", activation="relu"))
	model.add(layers.Dropout(0.5))
	model.add(layers.MaxPooling1D(pool_size=2))
	model.add(layers.Flatten())
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(1, activation="sigmoid"))
	model.summary()

	optimizer = tf.keras.optimizers.Adagrad(learning_rate=1e-3)
	model.compile(optimizer, loss='mse', metrics=['accuracy'])
	epocas = epochs
	results = model.fit(train_x, train_y, batch_size=128, epochs=epocas, verbose=1, validation_data=(test_x, test_y))

	plt.figure(3)
	plt.plot(np.arange(epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(epocas), results.history['accuracy'], 'k*-')
	plt.figure(4)
	plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(epocas), results.history['loss'], 'k*-')


(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=cant_de_palabras)

data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)


# TP4_4a(data[:20000], targets[:20000], 5)  # Original
TP4_4b(data[:2000], targets[:2000], 5)  # CNN
plt.show()
