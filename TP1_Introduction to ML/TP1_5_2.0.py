import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import mnist

import matplotlib.pyplot as plt


class Linear_classifier(object):
	def __init__(self, name):
		self.W = None
		self.name = name

	def accuracy(self, x, y):
		acc = np.zeros(y.shape[0])
		S = self.predict(x)
		b = y.T == S
		acc[b] = 1
		acc = np.sum(acc) / (y.shape[0])
		return acc

	def predict(self, x):
		score = self.score(x)
		return score.argmax(axis=1)

	def score(self, x):
		return np.dot(x, self.W)

	def loss(self, x, y):
		pass

	def loss_gradiente(self, x, y):
		pass

	def fit(self, X_train, Y_train, x_test, y_test, nbatch, learningRate, epochs, lam):

		im_shape = X_train.shape[1:]
		X = np.reshape(X_train, (X_train.shape[0], np.prod(im_shape)))  # cada fila es una imagen
		X = (np.hstack((np.ones((X_train.shape[0], 1)), X)))
		x_t = np.reshape(x_test, (x_test.shape[0], np.prod(im_shape)))  # cada fila es una imagen
		x_t = (np.hstack((np.ones((x_test.shape[0], 1)), x_t)))

		Y = Y_train.ravel().T
		y_t = y_test.ravel().T
		self.W = np.random.randn(np.prod(im_shape) + 1, np.amax(Y) + 1)

		nro_batchs = int(Y.shape[0] / nbatch)
		loss = np.zeros(epochs)
		accuracy = np.zeros(epochs)
		val_loss = np.zeros(epochs)
		val_acc = np.zeros(epochs)

		for i in range(epochs):
			indice = X_train.shape[0]
			indice = np.arange(indice)
			np.random.shuffle(indice)
			print('epoca: ' + str(i))
			acc = 0
			loss_train = 0
			test_acc = 0
			for j in range(nro_batchs):
				x_batch = X[indice[(j * nbatch):((j + 1) * nbatch)], :]
				y_batch = Y[indice[(j * nbatch):((j + 1) * nbatch)]]
				x_batch = x_batch.astype(np.int16)
				y_batch = y_batch.astype(np.int16)

				loss, gradiente = self.loss_grad(x_batch, y_batch, lam)
				self.W = self.W - learningRate * gradiente
				acc += self.accuracy(x_batch, y_batch)
				loss_train += loss
				test_acc += self.accuracy(x_t, y_t)

			val_loss[i] = loss_train / nro_batchs
			accuracy[i] = acc / nro_batchs
			val_acc[i] = test_acc / nro_batchs

		plt.figure()
		plt.plot(list(range(epochs)), val_loss, '*-')
		plt.title('Evolución de la Loss - {}'.format(self.name))
		plt.xlabel('Epocas')
		plt.ylabel('Loss')
		plt.ylim(0, 50000)
		plt.savefig('Loss{}.pdf'.format(self.name))
		plt.figure()
		plt.title('Evolución del Accuracy - .{}'.format(self.name))
		plt.xlabel('Epocas')
		plt.ylabel('Accuracy')
		plt.plot(list(range(epochs)), val_acc, 'r*-')
		plt.plot(list(range(epochs)), accuracy, 'k*-')
		plt.legend(['x_test', 'x_train'])
		plt.ylim(0, 100)

		plt.savefig('Accuracy{}.pdf'.format(self.name))

	def loss_grad(self, x_batch, y_batch, lam):
		pass


class SVM(Linear_classifier):

	def __init__(self, name):
		super().__init__(name)

	def loss_grad(self, x, y, lam):
		delta = 1.0
		r = (self.W * self.W).sum()
		scores = self.score(x)
		# print(y.T, np.arange(y.shape[0]))
		scores_correcto = scores[np.arange(x.shape[0]), y.T]
		margenes = scores - scores_correcto[:, np.newaxis] + delta
		margenes = np.maximum(0, margenes)
		margenes[np.arange(x.shape[0]), y] = 0
		loss = np.sum(margenes, axis=1)
		loss = np.mean(loss) + 0.5 * lam * r

		binario = margenes
		binario[binario > 0] = 1
		cuentas = np.sum(binario, axis=1)

		binario[np.arange(x.shape[0]), y] = - cuentas
		grad = np.dot(x.T, binario)
		grad = grad / x.shape[0]
		return loss, grad


class SoftMax(Linear_classifier):

	def __init__(self, name):
		super().__init__(name)

	def loss_grad(self, x_batch, y_batch, lam):
		r = np.sum(self.W * self.W)
		sc = self.score(x_batch)

		sc = sc - sc.max(axis=1)[:, np.newaxis]
		score_correcto = sc[np.arange(x_batch.shape[0]), y_batch]
		exp = np.exp(sc)
		sumaExp = np.sum(exp, axis=1)
		log = np.log(sumaExp) - score_correcto
		loss = log.mean() + lam * r / 2

		inversa = 1 / sumaExp
		grad = inversa[:, np.newaxis] * exp
		grad[np.arange(y_batch.shape[0]), y_batch] -= 1

		grad = np.dot(x_batch.T, grad)
		grad = grad / x_batch.shape[0]
		grad = grad + lam * self.W

		return loss, grad


print('Support Vector Machine method: MNIST')
(x_train, y_train), (x_test, y_test) = mnist.load_data()
model = SVM(name='SVM MNIST')
model.fit(x_train[:5000], y_train[:5000], x_test[:500], y_test[:500], nbatch=50, learningRate=1e-6, epochs=1000,
          lam=0.001)

print('SoftMax Method: MNIST')
(x_train, y_train), (x_test, y_test) = mnist.load_data()
model = SoftMax(name='SoftMax MNIST')
model.fit(x_train[:5000], y_train[:5000], x_test[:500], y_test[:500], nbatch=50, learningRate=1e-6, epochs=1000,
          lam=0.001)

print('Support Vector Machine method: CIFAR10')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
model = SVM(name='SVM CIFAR10')
model.fit(x_train[:5000], y_train[:5000], x_test[:500], y_test[:500], nbatch=50, learningRate=1e-6, epochs=1000,
          lam=0.001)

print('SoftMax Method: CIFAR10')
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
model = SoftMax(name='SoftMax cifar10')
model.fit(x_train[:5000], y_train[:5000], x_test[:500], y_test[:500], nbatch=50, learningRate=1e-6, epochs=1000,
          lam=0.001)
