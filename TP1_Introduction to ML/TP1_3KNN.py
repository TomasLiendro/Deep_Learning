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

	def predict(self, X):
		assert self.X is not None, 'Train method needs to be call first'
		Yp = np.zeros((X.shape[0], self.K), np.uint8)
		resu = np.zeros(X.shape[0], np.uint8)
		for idx in range(X.shape[0]):
			norm = np.linalg.norm(self.X - X[idx].ravel(), ord=2, axis=-1)
			ordenado_idsmin = norm.argsort()
			# print(norm[ordenado_idsmin[0]], norm[ordenado_idsmin[2]], norm[ordenado_idsmin[3]])
			idsmin = ordenado_idsmin[:self.K]
			for i in range(self.K):
				Yp[idx, i] = self.Y[idsmin[i]]
			b = mode(Yp[idx])

			resu[idx] = b[0]
		return resu


#
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print('x_train shape: ', x_train.shape)
# print(x_train.shape[0], 'train shape')
# print(x_test.shape[0], 'test shape')
# model = KNN(K=3)
# model.train(x_train, y_train)
# resu = model.predict(x_test[:20])
#
# print(resu)
# print(y_test[:20])

#
# (X_train, Y_train), (X_test, Y_test) = tf.datasets.mnist.load_data()
# 	# mnist.load_data()
# print('MNIST Dataset Shape:')
# print('X_train: ' + str(X_train.shape))
# print('Y_train: ' + str(Y_train.shape))
# print('X_test:  ' + str(X_test.shape))
# print('Y_test:  ' + str(Y_test.shape))
#
# model = KNN(K=3)
# model.train(X_train, Y_train)
# resu = model.predict(X_test[:20])
#
# print(resu)
# print(Y_test[:20])

# for i in range(len(X_test[:20])):
# 	sample = i
# 	image = X_test[sample]
# 	# plot the sample
# 	fig = plt.figure(i)
# 	plt.imshow(image, cmap='gray')
# plt.show()