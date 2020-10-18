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

	test_x = data[:1000]
	test_y = targets[:1000]
	train_x = data[1000:]
	train_y = targets[1000:]
	max_seq_l = 100
	test_x = pad_sequences(test_x, maxlen=max_seq_l, value=0.0)
	train_x = pad_sequences(train_x, maxlen=max_seq_l, value=0.0)

	model = models.Sequential()

	model.add(tf.keras.layers.Embedding(cant_de_palabras, 15, input_length=max_seq_l))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(200, activation="relu"))  # , input_shape=(train_x.shape[1],)))
	# model.add(layers.Dropout(0.5))
	# model.add(layers.Dense(25, activation="relu"))  # , input_shape=(train_x.shape[1],)))
	model.add(layers.MaxPooling1D(pool_size=2))
	model.add(layers.Flatten())
	model.add(layers.Dense(1, activation = "sigmoid"))
	model.summary()
	optimizer = tf.keras.optimizers.Adadelta(learning_rate=1)
	model.compile(optimizer, loss=losses.binary_crossentropy, metrics=["accuracy"])

	epocas = epochs
	results = model.fit(train_x, train_y, batch_size=128, epochs=epocas, verbose=2, validation_data=(test_x, test_y))

	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(epocas), results.history['accuracy'], 'k*-')
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('IMDB: Acc - Embeddings, densa')
	plt.xlabel('Época')
	plt.ylabel('Accuracy')
	plt.xlim([0, 20])
	plt.ylim([0, 1])
	plt.savefig('TP4_4aAcc.pdf')
	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(epocas), results.history['loss'], 'k*-')
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('IMDB: Loss - Embeddings, densa')
	plt.xlabel('Época')
	plt.ylabel('Loss')
	plt.xlim([0, 20])
	# plt.ylim([10, 50])
	plt.savefig('TP4_4aLoss.pdf')


def TP4_4b(data, targets, epochs):
	# data = vectorize(data)
	targets = np.array(targets).astype("float32")

	test_x = data[:1000]
	test_y = targets[:1000]
	train_x = data[1000:]
	train_y = targets[1000:]
	max_seq_l = 100
	test_x = pad_sequences(test_x, maxlen=max_seq_l, value=0.0)
	train_x = pad_sequences(train_x, maxlen=max_seq_l, value=0.0)

	model = models.Sequential()

	model.add(layers.Embedding(cant_de_palabras, 15, input_length=max_seq_l))
	model.add(layers.Dropout(0.5))
	# Hidden - Layers
	model.add(layers.Conv1D(filters=128, kernel_size=2, padding="same", activation="relu"))
	model.add(layers.Dropout(0.5))
	model.add(layers.MaxPooling1D(pool_size=2))
	model.add(layers.Flatten())
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(1, activation="sigmoid"))
	model.summary()

	optimizer = tf.keras.optimizers.Adadelta(learning_rate=1)
	model.compile(optimizer, loss=losses.binary_crossentropy, metrics=['accuracy'])
	epocas = epochs
	results = model.fit(train_x, train_y, batch_size=128, epochs=epocas, verbose=2, validation_data=(test_x, test_y))

	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(epocas), results.history['accuracy'], 'k*-')
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('IMDB: Acc - Embeddings, Conv.')
	plt.xlabel('Época')
	plt.ylabel('Accuracy')
	plt.xlim([0, 20])
	plt.ylim([0, 1])
	plt.savefig('TP4_4bAcc.pdf')
	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(epocas), results.history['loss'], 'k*-')
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('IMDB: Loss - Embeddings, conv')
	plt.xlabel('Época')
	plt.ylabel('Loss')
	plt.xlim([0, 20])
	# plt.ylim([10, 50])
	plt.savefig('TP4_4bLoss.pdf')


(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=cant_de_palabras)

data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)


TP4_4a(data[:20000], targets[:20000], 50)  # Original
# TP4_4b(data[:20000], targets[:20000], 50)  # CNN
plt.show()
