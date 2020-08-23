import numpy as np


class Noiser:
	def __init__(self, minV, maxV):
		self._minV = minV
		self._maxV = maxV

	def __call__(self, x):
		argumento = x
		if type(argumento) is float:
			return self.sum_pseudo(x)
		else:
			return argumento

	def sum_pseudo(self, valor):
		pseudo = np.random.uniform(self._minV, self._maxV)
		return valor + pseudo


noiser = Noiser(-0.5, 0.5)

g = np.vectorize(noiser)
array = [1, 1.3, 3.4, 5]

print(g(array))
