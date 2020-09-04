from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist
import tensorflow.keras as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode


class KNN:

	def __init__(self, K=1):
		self.im_shape = None
		self.X = None
		self.Y = None
		self.K = K

	def train(self, X, Y):
		self.im_shape = X.shape[1:]
		self.X = np.reshape(X, (X.shape[0], np.prod(self.im_shape)))
		self.Y = Y

	def acc(self, y1, y2):
		acc = np.zeros(len(y2))
		for i in range(len(y2)):
			if y2[i] == y1[i]:
				acc[i] = 1
		acc = np.sum(acc) / len(y2)
		return acc

	def predict(self, X):
		assert self.X is not None, 'Train method needs to be call first'
		X = X.astype(np.int16)
		Yp = np.zeros((X.shape[0], self.K), np.int16)
		resu = np.zeros(X.shape[0], np.int16)
		for idx in range(X.shape[0]):
			norm = np.linalg.norm(self.X - X[idx].ravel(), ord=2, axis=-1)
			ordenado_idsmin = norm.argsort()
			idsmin = ordenado_idsmin[:self.K]
			for i in range(self.K):
				Yp[idx, i] = self.Y[idsmin[i]]
			b = mode(Yp[idx])

			resu[idx] = b[0]
		return resu

# Uncomment this to check Ex. 3.
# Leave this commented to run Ex 4.
'''
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
xtr, ytr = x_train, y_train
xt, yt = x_test[:20], y_test[:20]

for k in [1, 3, 5, 7, 11, 13, 15]:
	model = KNN(K=k)
	model.train(xtr, ytr)
	resu = model.predict(xt)
	acc = model.acc(yt, resu)
	print('CIFAR-10: K={}, Accuracy={}'.format(k, acc))

(X_train, Y_train), (X_test, Y_test) = tf.datasets.mnist.load_data()
xtr, ytr = X_train, Y_train
xt, yt = X_test[:100], Y_test[:100]

for k in [1, 3, 5, 7, 11, 13, 15]:
	model = KNN(K=k)
	model.train(xtr, ytr)
	resu = model.predict(xt)
	acc = model.acc(yt, resu)
	print('MNIST: K={}, Accuracy={}'.format(k, acc))
'''