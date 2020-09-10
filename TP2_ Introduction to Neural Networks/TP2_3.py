import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.datasets.cifar10 as cifar10


def sigmoide(X):  # X es un vector
	exp = np.exp(-X)
	exp = 1 + exp
	y = 1 / exp
	return y


def accuracy(y_est, y_real):
	acc = np.zeros((y_est.shape[0], 1))
	Y_pred = np.argmax(y_est, axis=1)
	acc[Y_pred[:, np.newaxis] == y_real] = 1
	acc_sum = np.sum(acc, axis=0)
	y = acc_sum/y_est.shape[0]
	return y


def MSE(S2, yy_real):
	y = np.mean(np.sum((S2 - yy_real) ** 2, axis=1))
	return y


def gradMSE(S2, yy_real):
	y = 2 * (S2-yy_real)
	return y


def grad_sigmoide(x):
	y = sigmoide(x) * (1 - sigmoide(x))
	return y


nclases = 10  # salida
nintermedia = 100  # intermedia
batch_size = 128
n_epocas = 50
learning_rate = 1e-6
reg_lambda = 1e-3
n_train_data = 50000

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

im_shape = x_train.shape[1:]
xtr = np.reshape(x_train[:n_train_data], (x_train[:n_train_data].shape[0], np.prod(im_shape)))
x_mean = np.mean(xtr[:, np.newaxis], axis=0)

xtr = xtr - x_mean

xtr = np.hstack([np.ones((x_train[:n_train_data].shape[0], 1)), xtr])
ytr = y_train[:n_train_data]

yy_tr = np.zeros((ytr.shape[0], nclases))  # esto es un vector de train_data x 10
yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # vector de train_data x 10

nro_batchs = int(ytr.shape[0] / batch_size)

im_shape_test = x_test.shape[1:]
xt = np.reshape(x_test, (x_test.shape[0], np.prod(im_shape_test)))
xt = xt - x_mean
xt = np.hstack([np.ones((x_test.shape[0], 1)), xt])
yt = y_test

W1 = np.random.randn(xtr.shape[1], nintermedia) * 0.001  # 3073 x 100, meto el primer bias
W2 = np.random.randn(W1.shape[1]+1, nclases) * 0.001 # 101 x 10 : meto el segundo bias
train_acc = np.zeros((n_epocas, 1))
train_loss = np.zeros((n_epocas, 1))
test_acc = np.zeros((n_epocas, 1))

for i in range(n_epocas):
	indice = xtr.shape[0]
	indice = np.arange(indice)
	np.random.shuffle(indice)
	loss = 0
	for j in range(nro_batchs):
		# Forward
		x_batch = xtr[indice[(j * batch_size):((j + 1) * batch_size)], :]  # de batch_size x 3073
		y_batch = ytr[indice[(j * batch_size):((j + 1) * batch_size)]]  # de batch_size x 10

		yyy_tr = yy_tr[indice[(j * batch_size):((j + 1) * batch_size)], :]
		# Capa 1:
		Y1 = np.dot(x_batch, W1)  # esto es un vector de batch_size x 101
		S1 = sigmoide(Y1)
		S1_prima = np.hstack([np.ones((x_batch.shape[0], 1)), S1])  # esto es un vector de batch_size x 101
		# Capa 2
		S2 = np.dot(S1_prima, W2)  # es un vector de batch_size x 10

		# Calculo la regularización
		reg1 = np.sum(W1 * W1)
		reg2 = np.sum(W2 * W2)
		reg = reg1 + reg2
		# Cálculo de la Loss

	# 	Backward
		grad = gradMSE(S2, yyy_tr)
		grad_W2 = np.dot(S1_prima.T, grad)
		grad = np.dot(grad, W2.T)
		grad = grad[:, 1:]
		grad = grad_sigmoide(Y1) * grad
		grad_W1 = np.dot(x_batch.T, grad)

		W1 = W1 - learning_rate * (grad_W1 + reg * W1)
		W2 = W2 - learning_rate * (grad_W2 + reg * W2)

		train_loss[i] += MSE(S2, yyy_tr) + (reg1 + reg2) * 0.5
		train_acc[i] += accuracy(S2, y_batch)

	train_acc[i] = train_acc[i]/nro_batchs
	train_loss[i] = train_loss[i]/nro_batchs

	print('Epoca: '+str(i) + '  Accuracy: '+str(train_acc[i]))

plt.figure()
plt.plot(train_acc,'r*')
plt.figure()
plt.plot(train_loss,'b-*')
# plt.ylim(0,train_loss[1])
plt.show()
