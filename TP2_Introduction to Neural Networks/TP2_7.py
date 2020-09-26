import numpy as np
from itertools import product
from NN import layers, models, activations, optimizers, losses, regularizers

import matplotlib.pyplot as plt

def create_struct(x_train, y_train,neuronas2=1, problem_name='XOR'):
	# Create model: : 1st architecture
	model = models.Network()
	model.add(layers.Dense(units=neuronas2, activation=activations.Tanh(), input_dim=x_train.shape[1], factor=1e-1))
	model.add(layers.Dense(units=1, activation=activations.Tanh(),factor=1e-1))
	model.fit(x_train, y_train, test_data=None, epochs=10000, loss=losses.MSE(),
	          opt=optimizers.BGD(lr=0.01, bs=x_train.shape[0]), name=problem_name, reg=regularizers.L2(lam=1e-3))

	# Create model: 2nd architecture
	# model2 = models.Network()
	# model2.add(layers.Dense(units=2, activation=activations.Tanh(),input_dim=x_train.shape[1],factor=1e-1))
	# model2.add(layers.Concatenate(x_train.shape[1]))
	#
	# model2.add(layers.Dense(units=1, activation=activations.Tanh(),factor=1e-1))
	# model2.fit(x_train, y_train, test_data=None, epochs=10000, loss=losses.MSE(), opt=optimizers.BGD(lr=1e-4, bs=x_train.shape[0]), name=problem_name, reg=regularizers.L2(lam=1e-5))


# Barrido con N_prima fijo
N_prima = 15
for i in np.arange(0, 12, 3):
	N=1 + i
	x_train = np.array([i for i in product([-1, 1], repeat=N)])
	y_train = np.where(np.prod(x_train, axis=1) == 1, 1, -1)[:, np.newaxis]

	create_struct(x_train, y_train, neuronas2=N_prima)

# Barrido con N fijo
N = 2
for i in np.arange(0, 12, 3):
	N_prima = 1 + i
	x_train = np.array([i for i in product([-1, 1], repeat=N)])
	y_train = np.where(np.prod(x_train, axis=1) == 1, 1, -1)[:, np.newaxis]

	create_struct(x_train, y_train, neuronas2=N_prima)

# plt.figure(1)
# plt.legend(['N: 2 N\': 2', 'N: 2 N\': 3', 'N: 2 N\': 4', 'N: 2 N\': 5', 'N: 2 N\': 6', 'N: 2 N\': 7'])
# plt.savefig('TP2_7_acc.pdf')
#
# plt.figure(2)
# plt.legend(['N: 2 N\': 2', 'N: 2 N\': 3', 'N: 2 N\': 4', 'N: 2 N\': 5', 'N: 2 N\': 6', 'N: 2 N\': 7'])
# plt.savefig('TP2_7_loss.pdf')
#
# plt.figure(1)
# plt.legend(['N: 3 N\': 15', 'N: 6 N\': 15', 'N: 9 N\': 15', 'N: 12 N\': 15', 'N: 15 N\': 15'])
# plt.savefig('TP2_7_acc.pdf')
#
# plt.figure(2)
# plt.legend(['N: 3 N\': 15', 'N: 6 N\': 15', 'N: 9 N\': 15', 'N: 12 N\': 15', 'N: 15 N\': 15'])
# plt.savefig('TP2_7_loss.pdf')


# plt.figure(1)
# plt.legend(['N: 12 N\': 1'])
# plt.savefig('NvsNp_acc3.pdf')
#
# plt.figure(2)
# plt.legend(['N: 12 N\': 1'])
# plt.savefig('NvsNp_loss3.pdf')

# plt.show()

# print(x_train, y_train)
