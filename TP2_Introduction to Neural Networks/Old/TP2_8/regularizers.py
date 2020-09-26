import numpy as np


class Regularizer:
	def __init__(self, lam):
		self.lam = lam

	def __call__(self, W=None):
		pass

	def gradient(self, W=None):
		pass


class L1(Regularizer):
	def __init__(self, lam):
		super().__init__(lam)

	def __call__(self, W=None):
		return np.sum(np.abs(W)) * self.lam

	def gradient(self, lamda=0, W=None):
		# return np.ones(W.shape)
		return Exception, 'L1 Gradient not defined'


class L2(Regularizer):
	def __init__(self, lam):
		super().__init__(lam)

	def __call__(self, W=None):
		return np.sum(W ** 2) * self.lam / 2

	def gradient(self, W=None):
		return 2 * np.sum(W) * self.lam
