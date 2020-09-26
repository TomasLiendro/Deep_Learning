from NN import layers, models, activations, optimizers, losses, regularizers
import tensorflow.keras.datasets.cifar10 as cifar10
import tensorflow.keras.datasets.mnist as mnist
import matplotlib.pyplot as plt
import numpy as np


def create_struct(x_train, y_train, test_data=None, problem_name='CIFAR10'):
	# Creo primera NN
	model = models.Network()
	model.add(
		layers.Dense(units=100, activation=activations.Sigmoid(), input_dim=np.prod(x_train.shape[1:]), factor=1e-1))
	model.add(layers.Dense(units=100, activation=activations.Sigmoid(), factor=1e-1))
	model.add(layers.Dense(units=10, activation=activations.Linear(), factor=1e-1))
	model.fit(x_train, y_train, test_data=test_data, epochs=100, loss=losses.MSE(),
	          opt=optimizers.BGD(lr=1e-4, bs=64), name=problem_name, reg=regularizers.L2(lam=1e-3))

	# Creo segunda NN
	model2 = models.Network()
	model2.add(
		layers.Dense(units=100, activation=activations.Sigmoid(), input_dim=np.prod(x_train.shape[1:]), factor=1e-1))
	model2.add(layers.Dense(units=100, activation=activations.Sigmoid(), factor=1e-2))
	model2.add(layers.Dense(units=10, activation=activations.Linear(), factor=1e-1))
	model2.fit(x_train, y_train, test_data=test_data, epochs=100, loss=losses.SVM(),
	          opt=optimizers.BGD(lr=1e-4, bs=64), name=problem_name, reg=regularizers.L2(lam=1e-3))

	# Creo tercera NN
	model3 = models.Network()
	model3.add(
		layers.Dense(units=100, activation=activations.Sigmoid(), input_dim=np.prod(x_train.shape[1:]), factor=1e-1))
	model3.add(layers.Dense(units=100, activation=activations.Sigmoid(), factor=1e-2))
	model3.add(layers.Dense(units=10, activation=activations.Linear(), factor=1e-1))
	model3.fit(x_train, y_train, test_data=test_data, epochs=100, loss=losses.CCE(),
	          opt=optimizers.BGD(lr=1e-4, bs=64), name=problem_name, reg=regularizers.L2(lam=1e-3))



# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

yy_tr = np.zeros((y_train.shape[0], np.max(y_train) + 1))  # esto es un vector de train_data x 10
yy_tr[np.arange(y_train.shape[0]), y_train.T] = 1  # vector de train_data x 10

yy_test = np.zeros((y_test.shape[0], np.max(y_test) + 1))  # esto es un vector de train_data x 10
yy_test[np.arange(y_test.shape[0]), y_test.T] = 1  # vector de train_data x 10

create_struct(x_train[:5000, :], yy_tr[:5000, :], test_data=(x_test[:500, :], yy_test[:500, :]), problem_name='CIFAR10')
