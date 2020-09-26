import numpy as np


class Activations:
	def __call__(self, x):
		pass

	def gradient(self, x):
		pass


class Relu(Activations):
	def __call__(self, x):
		return np.maximum(0, x)

	def gradient(self, x):
		return np.where(x < 0, 0, 1)


class Tanh(Activations):
	def __call__(self, x):
		return np.tanh(x)

	def gradient(self, x):
		return 1 - self.__call__(x) ** 2


class Sigmoid(Activations):
	def __call__(self, x):
		exp = 1 + np.exp(-x)
		return 1 / exp

	def gradient(self, x):
		return self.__call__(x) * (1 - self.__call__(x))


class Linear(Activations):
	def __call__(self, x):
		return x

	def gradient(self, x):
		return np.ones(x.shape)
