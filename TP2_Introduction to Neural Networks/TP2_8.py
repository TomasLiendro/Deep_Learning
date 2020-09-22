from TP6 import layers, models, activations, optimizers, losses, regularizers
import tensorflow.keras.datasets.cifar10 as cifar10
import numpy as np


def create_struct(x_train, y_train, test_data=None, problem_name='CIFAR10'):
	model = models.Network()
	model.add(layers.Dense(units=100, activation=activations.Sigmoid(), input_dim=np.prod(x_train.shape[1:])))
	model.add(layers.Dense(units=100, activation=activations.Sigmoid()))
	model.add(layers.Dense(units=10, activation=activations.Linear()))
	model.fit(x_train, y_train, test_data=test_data, epochs=200, loss=losses.MSE(),
	          opt=optimizers.BGD(lr=1e-4, bs=128), name=problem_name, reg=regularizers.L2(lam=1e-3))


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

yy_tr = np.zeros((y_train.shape[0], np.max(y_train) + 1))  # esto es un vector de train_data x 10
yy_tr[np.arange(y_train.shape[0]), y_train.T] = 1  # vector de train_data x 10

create_struct(x_train[:2000,:], yy_tr[:2000,:],test_data=None , problem_name='CIFAR10')
