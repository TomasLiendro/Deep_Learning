import tensorflow.keras.datasets.cifar10 as cifar10
import tensorflow.keras.datasets.cifar100 as cifar100
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations
import tensorflow.keras.regularizers as regularizers
import tensorflow.keras.losses as losses
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import ImageDataGenerator


import numpy as np

#  AlexNet


def ANet_C10(x_train, y_train, x_test, y_test, size):
	x_train = x_train.astype("float32") / 255
	x_test = x_test.astype("float32") / 255

	# Augmentation
	datagen = ImageDataGenerator(
		rotation_range=90,
		horizontal_flip=True,
		width_shift_range=0.1,
		height_shift_range=0.1
	)
	datagen.fit(x_train)
	# acondiciono los datos de training
	im_shape = x_train.shape[1:]
	xtr = np.reshape(x_train, (x_train.shape[0], np.prod(im_shape)))  # Los datos de train como vector

	x_mean = np.mean(xtr, axis=0)[np.newaxis, :]  # media de los datos de training
	std = np.std(xtr, axis=0)[np.newaxis, :]  # STD de los datos de training

	xtr = (xtr - x_mean)

	ytr = y_train
	yy_tr = np.zeros((ytr.shape[0], 10))  # esto es un vector de train_data x 10 para representar las clases
	yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # vector de train_data x 10

	# Acondiciono los datos de testing:
	im_shape_test = x_test.shape[1:]
	xt = np.reshape(x_test, (x_test.shape[0], np.prod(im_shape_test)))  # Los datos de test como vector

	xt = (xt - x_mean)
	yt = y_test
	yy_t = np.zeros((yt.shape[0], 10))  # esto es un vector de train_data x 10 para representar las clases
	yy_t[np.arange(yt.shape[0]), yt.T] = 1  # vector de train_data x 10

	xtr = np.reshape(xtr, (-1, im_shape[0], im_shape[1], im_shape[2]))
	xt = np.reshape(xt, (-1, im_shape_test[0], im_shape_test[1], im_shape_test[2]))

	model = models.Model()
	input_img = layers.Input(shape=xtr.shape[1:])
	x = layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu')(input_img)
	x = layers.MaxPool2D(pool_size=(3, 3), strides=(2,2))(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu')(x)
	x = layers.MaxPool2D(pool_size=(3, 3), strides=(2,2))(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu')(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu')(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu')(x)
	x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
	x = layers.Flatten()(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(100, activation='relu')(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(50, activation='relu')(x)
	out = layers.Dense(yy_tr.shape[1], activation=activations.softmax)(x)

	model = models.Model(input_img, out)

	model.summary()
	lossCCE = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

	optimizer = tf.keras.optimizers.Adagrad(learning_rate=.01)
	model.compile(optimizer, loss=losses.binary_crossentropy, metrics=['accuracy'])
	epocas = 25
	results = model.fit(xtr, yy_tr, batch_size=128, epochs=epocas, verbose=1, validation_data=(xt, yy_t))

	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(epocas), results.history['accuracy'], 'k*-')
	# plt.legend(['Training data', 'Testing data'], loc='lower right', shadow=False)
	plt.ylabel("Accuracy")
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('AlexNet, CIFAR10: Accuracy')
	plt.savefig('TP4_10_C10_anet_acc.pdf')

	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(epocas), results.history['loss'], 'k*-')
	plt.xlabel("Épocas")
	plt.ylabel("Loss")
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('AlexNet, CIFAR10: Loss')
	plt.savefig('TP4_10_C10_anet_loss.pdf')


def VGG16_C10(x_train, y_train, x_test, y_test, size):
	x_train = x_train.astype("float32") / 255
	x_test = x_test.astype("float32") / 255

	# Augmentation
	datagen = ImageDataGenerator(
		rotation_range=90,
		horizontal_flip=True,
		width_shift_range=0.1,
		height_shift_range=0.1
	)
	datagen.fit(x_train)
	# acondiciono los datos de training
	im_shape = x_train.shape[1:]
	xtr = np.reshape(x_train, (x_train.shape[0], np.prod(im_shape)))  # Los datos de train como vector

	x_mean = np.mean(xtr, axis=0)[np.newaxis, :]  # media de los datos de training
	std = np.std(xtr, axis=0)[np.newaxis, :]  # STD de los datos de training

	xtr = (xtr - x_mean)

	ytr = y_train
	yy_tr = np.zeros((ytr.shape[0], 10))  # esto es un vector de train_data x 10 para representar las clases
	yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # vector de train_data x 10

	# Acondiciono los datos de testing:
	im_shape_test = x_test.shape[1:]
	xt = np.reshape(x_test, (x_test.shape[0], np.prod(im_shape_test)))  # Los datos de test como vector

	xt = (xt - x_mean)
	yt = y_test
	yy_t = np.zeros((yt.shape[0], 10))  # esto es un vector de train_data x 10 para representar las clases
	yy_t[np.arange(yt.shape[0]), yt.T] = 1  # vector de train_data x 10

	xtr = np.reshape(xtr, (-1, im_shape[0], im_shape[1], im_shape[2]))
	xt = np.reshape(xt, (-1, im_shape_test[0], im_shape_test[1], im_shape_test[2]))

	model = models.Model()
	input_img = layers.Input(shape=xtr.shape[1:])
	x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(input_img)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.3)(x)
	x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPool2D(pool_size=(2, 2), strides=(1,1))(x)
	x = layers.Dropout(0.2)(x)

	x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.4)(x)
	x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPool2D(pool_size=(2, 2), strides=(1,1))(x)

	x = layers.Dropout(0.4)(x)
	x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPool2D(pool_size=(3, 3), strides=(2,2))(x)

	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)


	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)

	x = layers.Flatten()(x)
	x = layers.Dropout(0.2)(x)

	x = layers.Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dense(250, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dense(250, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
	x = layers.BatchNormalization()(x)

	out = layers.Dense(yy_tr.shape[1], activation=activations.softmax, kernel_regularizer=regularizers.l2(0.01))(x)

	model = models.Model(input_img, out)

	model.summary()
	lossCCE = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

	optimizer = tf.keras.optimizers.Adagrad(learning_rate=1)
	model.compile(optimizer, loss=losses.binary_crossentropy, metrics=['accuracy'])
	epocas = 25
	results = model.fit(xtr, yy_tr, batch_size=128, epochs=epocas, verbose=1, validation_data=(xt, yy_t))

	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(epocas), results.history['accuracy'], 'k*-')
	# plt.legend(['Training data', 'Testing data'], loc='lower right', shadow=False)
	plt.ylabel("Accuracy")
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('VGG16, CIFAR10: Accuracy')
	plt.savefig('TP4_10_C10_vgg_acc.pdf')

	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(epocas), results.history['loss'], 'k*-')
	plt.xlabel("Épocas")
	plt.ylabel("Loss")
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('VGG16, CIFAR10: Loss')
	plt.savefig('TP4_10_C10_vgg16_loss.pdf')



def ANet_C100(x_train, y_train, x_test, y_test, size):
	x_train = x_train.astype("float32") / 255
	x_test = x_test.astype("float32") / 255

	# Augmentation
	datagen = ImageDataGenerator(
		rotation_range=90,
		horizontal_flip=True,
		width_shift_range=0.1,
		height_shift_range=0.1
	)
	datagen.fit(x_train)
	# acondiciono los datos de training
	im_shape = x_train.shape[1:]
	xtr = np.reshape(x_train, (x_train.shape[0], np.prod(im_shape)))  # Los datos de train como vector

	x_mean = np.mean(xtr, axis=0)[np.newaxis, :]  # media de los datos de training
	std = np.std(xtr, axis=0)[np.newaxis, :]  # STD de los datos de training

	xtr = (xtr - x_mean)

	ytr = y_train
	yy_tr = np.zeros((ytr.shape[0], 100))  # esto es un vector de train_data x 10 para representar las clases
	yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # vector de train_data x 10

	# Acondiciono los datos de testing:
	im_shape_test = x_test.shape[1:]
	xt = np.reshape(x_test, (x_test.shape[0], np.prod(im_shape_test)))  # Los datos de test como vector

	xt = (xt - x_mean)
	yt = y_test
	yy_t = np.zeros((yt.shape[0], 100))  # esto es un vector de train_data x 10 para representar las clases
	yy_t[np.arange(yt.shape[0]), yt.T] = 1  # vector de train_data x 10

	xtr = np.reshape(xtr, (-1, im_shape[0], im_shape[1], im_shape[2]))
	xt = np.reshape(xt, (-1, im_shape_test[0], im_shape_test[1], im_shape_test[2]))

	model = models.Model()
	input_img = layers.Input(shape=xtr.shape[1:])
	x = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu')(input_img)
	x = layers.MaxPool2D(pool_size=(3, 3), strides=(2,2))(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu')(x)
	x = layers.MaxPool2D(pool_size=(3, 3), strides=(2,2))(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu')(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu')(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu')(x)
	x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
	x = layers.Flatten()(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(1000, activation='relu')(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Dense(500, activation='relu')(x)
	x = layers.Dropout(0.2)(x)
	out = layers.Dense(yy_tr.shape[1], activation=activations.softmax)(x)

	model = models.Model(input_img, out)

	model.summary()
	lossCCE = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

	optimizer = tf.keras.optimizers.Adagrad(learning_rate=.01)
	model.compile(optimizer, loss=losses.binary_crossentropy, metrics=['accuracy'])
	epocas = 25
	results = model.fit(xtr, yy_tr, batch_size=128, epochs=epocas, verbose=1, validation_data=(xt, yy_t))

	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(epocas), results.history['accuracy'], 'k*-')
	# plt.legend(['Training data', 'Testing data'], loc='lower right', shadow=False)
	plt.ylabel("Accuracy")
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('AlexNet, CIFAR100: Accuracy')
	plt.savefig('TP4_10_C100_anet_acc.pdf')

	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(epocas), results.history['loss'], 'k*-')
	plt.xlabel("Épocas")
	plt.ylabel("Loss")
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('AlexNet, CIFAR100: Loss')
	plt.savefig('TP4_10_C100_anet_loss.pdf')


def VGG16_C100(x_train, y_train, x_test, y_test, size):
	x_train = x_train.astype("float32") / 255
	x_test = x_test.astype("float32") / 255

	# Augmentation
	datagen = ImageDataGenerator(
		rotation_range=90,
		horizontal_flip=True,
		width_shift_range=0.1,
		height_shift_range=0.1
	)
	datagen.fit(x_train)
	# acondiciono los datos de training
	im_shape = x_train.shape[1:]
	xtr = np.reshape(x_train, (x_train.shape[0], np.prod(im_shape)))  # Los datos de train como vector

	x_mean = np.mean(xtr, axis=0)[np.newaxis, :]  # media de los datos de training
	std = np.std(xtr, axis=0)[np.newaxis, :]  # STD de los datos de training

	xtr = (xtr - x_mean)

	ytr = y_train
	yy_tr = np.zeros((ytr.shape[0], 100))  # esto es un vector de train_data x 10 para representar las clases
	yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # vector de train_data x 10

	# Acondiciono los datos de testing:
	im_shape_test = x_test.shape[1:]
	xt = np.reshape(x_test, (x_test.shape[0], np.prod(im_shape_test)))  # Los datos de test como vector

	xt = (xt - x_mean)
	yt = y_test
	yy_t = np.zeros((yt.shape[0], 100))  # esto es un vector de train_data x 10 para representar las clases
	yy_t[np.arange(yt.shape[0]), yt.T] = 1  # vector de train_data x 10

	xtr = np.reshape(xtr, (-1, im_shape[0], im_shape[1], im_shape[2]))
	xt = np.reshape(xt, (-1, im_shape_test[0], im_shape_test[1], im_shape_test[2]))

	model = models.Model()
	input_img = layers.Input(shape=xtr.shape[1:])
	x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(input_img)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.3)(x)
	x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPool2D(pool_size=(2, 2), strides=(1,1))(x)
	x = layers.Dropout(0.2)(x)

	x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.4)(x)
	x = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPool2D(pool_size=(2, 2), strides=(1,1))(x)

	x = layers.Dropout(0.4)(x)
	x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPool2D(pool_size=(3, 3), strides=(2,2))(x)

	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.MaxPool2D(pool_size=(2, 2), strides=(2,2))(x)


	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.2)(x)
	x = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu', kernel_regularizer=regularizers.l2(0.0001))(x)
	x = layers.BatchNormalization()(x)

	x = layers.Flatten()(x)
	x = layers.Dropout(0.2)(x)

	x = layers.Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dense(500, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dense(250, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
	x = layers.BatchNormalization()(x)
	x = layers.Dropout(0.5)(x)
	out = layers.Dense(yy_tr.shape[1], activation=activations.softmax, kernel_regularizer=regularizers.l2(0.01))(x)

	model = models.Model(input_img, out)

	model.summary()
	lossCCE = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

	optimizer = tf.keras.optimizers.Adagrad(learning_rate=1)
	model.compile(optimizer, loss=losses.binary_crossentropy, metrics=['accuracy'])
	epocas = 25
	results = model.fit(xtr, yy_tr, batch_size=128, epochs=epocas, verbose=1, validation_data=(xt, yy_t))

	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_accuracy'], 'r*-')
	plt.plot(np.arange(epocas), results.history['accuracy'], 'k*-')
	# plt.legend(['Training data', 'Testing data'], loc='lower right', shadow=False)
	plt.ylabel("Accuracy")
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('VGG16, CIFAR100: Accuracy')
	plt.savefig('TP4_10_C100_vgg_acc.pdf')

	plt.figure()
	plt.plot(np.arange(epocas), results.history['val_loss'], 'r*-')
	plt.plot(np.arange(epocas), results.history['loss'], 'k*-')
	plt.xlabel("Épocas")
	plt.ylabel("Loss")
	plt.legend(['Datos de validación', 'Datos de entrenamiento'])
	plt.title('VGG16, CIFAR100: Loss')
	plt.savefig('TP4_10_C100_vgg16_loss.pdf')


# (x_train, y_train), (x_test, y_test) = cifar10.load_data()  # Cargo los datos de CIFAR-10
# size = x_train.shape[0]
# ANet_C10(x_train[:size], y_train[:size], x_test[:int(size / 10)], y_test[:int(size / 10)], size)
# VGG16_C10(x_train[:size], y_train[:size], x_test[:int(size / 10)], y_test[:int(size / 10)], size)


(x_train, y_train), (x_test, y_test) = cifar100.load_data()  # Cargo los datos de CIFAR-100
size = x_train.shape[0]
ANet_C100(x_train[:size], y_train[:size], x_test[:int(size / 10)], y_test[:int(size / 10)], size)
VGG16_C100(x_train[:size], y_train[:size], x_test[:int(size / 10)], y_test[:int(size / 10)], size)
