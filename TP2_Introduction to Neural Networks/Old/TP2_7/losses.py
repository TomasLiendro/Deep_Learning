import numpy as np


class Loss:
	def __call__(self, y_sc, y_real):
		pass

	def gradient(self, y_sc, y_reals):
		pass


class MSE(Loss):
	def __call__(self, y_sc, y_real):
		loss = np.mean(np.sum((y_sc - y_real) ** 2, axis=1))
		return loss

	def gradient(self, y_sc, y_real):
		grad = 2 * (y_sc - y_real)
		return grad
