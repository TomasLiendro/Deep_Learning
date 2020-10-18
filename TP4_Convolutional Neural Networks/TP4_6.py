import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.regularizers as regularizer
import tensorflow.keras.losses as losses
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
import numpy as np

full_data = np.genfromtxt('pima-indians-diabetes.csv',delimiter=',')

data = full_data[:, :-1]
targets = full_data[:, -1]


def TP4_6a():
	kf = KFold(n_splits=5, shuffle=True)
	acc = []
	val_acc = []
	loss = []
	val_loss = []
	epocas = 50

	for train_index, test_index in kf.split(data):
		train_x, test_x = data[train_index], data[test_index]
		train_y, test_y = targets[train_index], targets[test_index]

		model = models.Sequential()
		model.add(layers.Dense(250, activation="relu", input_shape=(train_x.shape[1],)))
		#model.add(layers.Dense(100, activation="relu"))
		model.add(layers.Dense(100, activation="relu"))
		model.add(layers.Dense(50, activation="relu"))
		model.add(layers.Dense(50, activation="relu"))
		model.add(layers.Dense(50, activation="relu"))
		model.add(layers.Dense(1, activation="sigmoid"))
		model.summary()
		optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
		model.compile(optimizer, loss=tf.keras.losses.binary_crossentropy, metrics=["accuracy"])
		results = model.fit(train_x, train_y, batch_size=128, epochs=epocas, verbose=2,
							validation_data=(test_x, test_y))
		acc.append(results.history['accuracy'])
		val_acc.append(results.history['val_accuracy'])
		val_loss.append(results.history['val_loss'])
		loss.append(results.history['loss'])

	minLoss = np.min(val_loss, axis=0)
	maxLoss = np.max(val_loss, axis=0)
	meanLoss = np.mean(val_loss, axis=0)
	minAcc = np.min(val_acc, axis=0)
	maxAcc = np.max(val_acc, axis=0)
	meanAcc = np.mean(val_acc, axis=0)

	plt.figure()
	plt.fill_between(np.arange(epocas), minLoss, maxLoss, alpha=0.5, facecolor="red")
	plt.plot(meanLoss, color="red")
	plt.title("Ejercicio 6: Loss")
	plt.ylabel("Loss")
	plt.xlabel("Épocas")
	# plt.show()
	plt.xlim([0, epocas])
	plt.savefig('TP4_6_a_loss.pdf')

	plt.figure()
	plt.fill_between(np.arange(epocas), minAcc, maxAcc, alpha=0.5, facecolor="blue")
	plt.plot(meanAcc, color="blue")
	plt.title("Ejercicio 6: Accuracy")
	plt.ylabel("Accuracy")
	plt.xlabel("Épocas")
	plt.ylim([0,1])
	plt.xlim([0, epocas])
	plt.savefig('TP4_6_a_acc.pdf')
	plt.figure(3)
	plt.plot(np.arange(epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(epocas), results.history['accuracy'], 'k*-')
	plt.figure(4)
	plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(epocas), results.history['loss'], 'k*-')

# def TP4_6b():
# 	kf = KFold(n_splits=5, shuffle=True)
# 	acc = []
# 	val_acc = []
# 	loss = []
# 	val_loss = []
# 	epocas = 20
# 	for train_index, test_index in kf.split(data):
# 		train_x, test_x = data[train_index], data[test_index]
# 		train_y, test_y = targets[train_index], targets[test_index]
#
# 		model = models.Model()
# 		input = layers.Input(shape=[8])
# 		conv1 = layers.Conv1D(filters=2, kernel_size=2, padding="same", activation="relu")(input)
# 		conv1 = layers.Conv1D(filters=2, kernel_size=2, padding="same", activation="relu")(conv1)
# 		fl1 = layers.Flatten()(conv1)
# 		out = layers.Dense(1, activation="sigmoid")(fl1)
# 		model = tf.keras.models.Model(input, out)
# 		model.summary()
#
# 		optimizer = tf.keras.optimizers.Adam(learning_rate=.0001)
#
# 		model.compile(optimizer, loss='binary_crossentropy', metrics=["accuracy"])
#
# 		results = model.fit(train_x, train_y, batch_size=16, epochs=epocas, verbose=1, validation_data=(test_x, test_y))
# 		acc.append(results.history['accuracy'])
# 		val_acc.append(results.history['val_accuracy'])
# 		val_loss.append(results.history['val_loss'])
# 		loss.append(results.history['loss'])
#
# 	minLoss = np.min(val_loss, axis=0)
# 	maxLoss = np.max(val_loss, axis=0)
# 	meanLoss = np.mean(val_loss, axis=0)
# 	minAcc = np.min(val_acc, axis=0)
# 	maxAcc = np.max(val_acc, axis=0)
# 	meanAcc = np.mean(val_acc, axis=0)
#
# 	plt.figure()
# 	plt.fill_between(np.arange(epocas), minLoss, maxLoss, alpha=0.5, facecolor="red")
# 	plt.plot(meanLoss, color="red")
# 	plt.title("Ejercicio 6: Loss")
# 	plt.ylabel("Loss")
# 	plt.xlabel("Épocas")
# 	plt.savefig('TP4_6_b_loss.pdf')
#
# 	plt.figure()
# 	plt.fill_between(np.arange(epocas), minAcc, maxAcc, alpha=0.5, facecolor="blue")
# 	plt.plot(meanAcc, color="blue")
# 	plt.title("Ejercicio 6: Accuracy")
# 	plt.ylabel("Accuracy")
# 	plt.xlabel("Épocas")
# 	plt.savefig('TP4_6_b_acc.pdf')


def TP4_6c():
	valores_medios = np.mean(data, axis=0)[0:]
	for i in range(data.shape[0]):
		for j in range(1, data.shape[1]):
			data[i, j] = np.where(data[i, j] == 0.0, valores_medios[j], data[i,j])

	kf = KFold(n_splits=5, shuffle=True)
	acc = []
	val_acc = []
	loss = []
	val_loss = []
	epocas = 50

	for train_index, test_index in kf.split(data):
		train_x, test_x = data[train_index], data[test_index]
		train_y, test_y = targets[train_index], targets[test_index]

		model = models.Sequential()
		model.add(layers.Dense(250, activation="relu", input_shape=(train_x.shape[1],)))
		#model.add(layers.Dense(100, activation="relu"))
		model.add(layers.Dense(100, activation="relu"))
		model.add(layers.Dense(50, activation="relu"))
		model.add(layers.Dense(50, activation="relu"))
		model.add(layers.Dense(50, activation="relu"))
		model.add(layers.Dense(1, activation="sigmoid"))
		model.summary()
		optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
		model.compile(optimizer, loss=tf.keras.losses.binary_crossentropy, metrics=["accuracy"])
		results = model.fit(train_x, train_y, batch_size=128, epochs=epocas, verbose=2, validation_data=(test_x, test_y))
		acc.append(results.history['accuracy'])
		val_acc.append(results.history['val_accuracy'])
		val_loss.append(results.history['val_loss'])
		loss.append(results.history['loss'])

	minLoss = np.min(val_loss, axis=0)
	maxLoss = np.max(val_loss, axis=0)
	meanLoss = np.mean(val_loss, axis=0)
	minAcc = np.min(val_acc, axis=0)
	maxAcc = np.max(val_acc, axis=0)
	meanAcc = np.mean(val_acc, axis=0)

	plt.figure()
	plt.fill_between(np.arange(epocas), minLoss, maxLoss, alpha=0.5, facecolor="red")
	plt.plot(meanLoss, color="red")
	plt.title("Ejercicio 6: Loss")
	plt.ylabel("Loss")
	plt.xlabel("Épocas")
	# plt.show()
	plt.xlim([0, epocas])
	plt.savefig('TP4_6_c_loss.pdf')

	plt.figure()
	plt.fill_between(np.arange(epocas), minAcc, maxAcc, alpha=0.5, facecolor="blue")
	plt.plot(meanAcc, color="blue")
	plt.title("Ejercicio 6: Accuracy")
	plt.ylabel("Accuracy")
	plt.xlabel("Épocas")
	plt.ylim([0,1])
	plt.xlim([0, epocas])
	plt.savefig('TP4_6_c_acc.pdf')
	plt.figure(3)
	plt.plot(np.arange(epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(epocas), results.history['accuracy'], 'k*-')

	plt.figure(4)
	plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(epocas), results.history['loss'], 'k*-')

TP4_6a()
# TP4_6b()
TP4_6c()
plt.show()

