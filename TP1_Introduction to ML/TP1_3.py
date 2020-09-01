from keras.datasets import cifar10
import numpy as np
from tensorflow.keras.datasets import mnist


class NN:

	def __init__(self):
		self.im_shape = None
		self.X = None
		self.Y = None

	def train(self, X, Y):
		self.im_shape = X.shape[1:]
		self.X = np.reshape(X, (X.shape[0], np.prod(self.im_shape)))
		self.Y = Y

	def predict(self, X):
		assert self.X is not None, 'Train method needs to be call first'
		Yp = np.zeros(X.shape[0], np.int16)
		X = X.astype(np.int16)
		for idx in range(X.shape[0]):
			norm = np.linalg.norm(self.X - X[idx].ravel(), axis=-1)
			idmin = np.argmin(norm)
			Yp[idx] = self.Y[idmin]
		return Yp


# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print('x_train shape: ', x_train.shape)
# print(x_train.shape[0], 'train shape')
# print(x_test.shape[0], 'test shape')
#
# model = NN()
# model.train(x_train, y_train)
# yp = model.predict(x_test[:10])


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print('MNIST Dataset Shape:')
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(Y_train.shape))
print('X_test:  ' + str(X_test.shape))
print('Y_test:  ' + str(Y_test.shape))

model = NN()
model.train(X_train, Y_train)
resu = model.predict(X_test[:20])

# print(resu)
# print(Y_test)
# acc = np.mean(resu == Y_test[:20])
# print('accuracy: %f' % acc)
