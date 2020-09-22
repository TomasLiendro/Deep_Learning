from TP6 import layers, models, activations, optimizers, losses, regularizers
import numpy as np

# Identify the problem: 'XOR' or 'Image'
problem_name = 'XOR'

# Dataset
x_train = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
y_train = np.array([[1], [-1], [-1], [1]])

# Create model: : 1st architecture
# model = models.Network()
# model.add(layers.Dense(units=2, activation=activations.Tanh(),input_dim=x_train.shape[1]))
# model.add(layers.Dense(units=1, activation=activations.Tanh()))

# Train Network
# model.fit(x_train, y_train, test_data=None, epochs=10000, loss=losses.MSE(), opt=optimizers.BGD(lr=0.01, bs=x_train.shape[0]), name=problem_name)

# Create model: 2nd architecture
model2 = models.Network()
model2.add(layers.Dense(units=2, activation=activations.Tanh(),input_dim=x_train.shape[1]))
model2.add(layers.Concatenate(x_train.shape[1]))

model2.add(layers.Dense(units=1, activation=activations.Tanh()))

# Train Network
model2.fit(x_train, y_train, test_data=None, epochs=10000, loss=losses.MSE(), opt=optimizers.BGD(lr=0.01, bs=x_train.shape[0]), name=problem_name, reg=regularizers.L2(lam=1e-3))
