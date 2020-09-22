import numpy as np


class Optimizer:
	def __init__(self, lr):
		self.lr = lr

	def __call__(self, X, Y, model):
		pass

	def update_W(self, W, gradW):
		pass


class BGD(Optimizer):
	def __init__(self, lr, bs):
		super().__init__(lr)
		self.bs = bs

	def __call__(self, X, Y, model):
		n_batches = int(Y.shape[0] / self.bs)
		indice = X.shape[0]
		indice = np.arange(indice)
		np.random.shuffle(indice)
		for j in range(n_batches):
			x_batch = X[indice[(j * self.bs):((j + 1) * self.bs)], :]
			y_batch = Y[indice[(j * self.bs):((j + 1) * self.bs)]]
			model.backward(x_batch, y_batch)  # Check

		model.acc_train[model.epoca] /= n_batches
		model.loss_train[model.epoca] /= n_batches

	def update_W(self, W, gradW):
		return W - self.lr * gradW
