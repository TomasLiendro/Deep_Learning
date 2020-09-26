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


class CCE(Loss):
	def __call__(self, y_sc, y_real):
		y_real = np.argmax(y_real, axis=1)
		y = np.max(y_sc.T, axis=0)
		y_pred = y_sc.T - y
		y_pred = y_pred.T
		real = y_pred[np.arange(y_real.shape[0]), y_real]
		L = -real + np.log(np.sum(np.exp(y_pred), axis=1))
		loss = np.mean(L)
		return loss

	def gradient(self, y_sc, y_real):
		y_real = np.argmax(y_real, axis=1)
		y = np.max(y_sc.T, axis=0)
		y_pred = y_sc.T - y
		y_pred = y_pred.T
		inter = np.sum(np.exp(y_pred), axis=1)
		margins = (np.exp(y_pred)).T / (inter)
		margins = margins.T
		margins[np.arange(y_pred.shape[0]), y_real] -= 1
		return margins


class SVM(Loss):
	def __call__(self, y_sc, y_real):
		y_real = np.argmax(y_real, axis=1)
		y_pred = y_sc.T
		delta = 1
		real = y_pred[y_real, np.arange(y_real.shape[0])]
		margins = y_pred - real + delta
		margins[margins < 0] = 0
		margins[y_real, np.arange(y_real.shape[0])] = 0
		L = np.sum(margins, axis=0)
		return np.mean(L)

	def gradient(self, y_sc, y_real):
		delta = 1
		y_real = np.argmax(y_real, axis=1)
		y_pred = y_sc.T
		real = y_pred[y_real, np.arange(y_real.shape[0])]
		margins = y_pred - real + delta
		margins[margins < 0] = 0
		margins[y_real, np.arange(y_real.shape[0])] = 0
		margins[margins > 0] = 1
		margins[y_real, np.arange(y_real.shape[0])] = -np.sum(margins, axis=0)
		return margins.T