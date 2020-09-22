import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.datasets.cifar10 as cifar10


def sigmoide(X):
	exp = np.exp(-X)
	exp = 1 + exp
	y = 1 / exp
	return y


def accuracy(y_est, y_real):
	acc = np.zeros((y_est.shape[0], 1))
	Y_pred = np.argmax(y_est, axis=1)
	acc[Y_pred[:, np.newaxis] == y_real] = 1
	acc_sum = np.sum(acc, axis=0)
	y = acc_sum / y_est.shape[0]
	return y


def loss(S2, yy_real, tipo='MSE'):
	if tipo == 'MSE':
		L = np.mean(np.sum((S2 - yy_real) ** 2, axis=1))
		grad_MSE = 2 * (S2 - yy_real)
		return grad_MSE, L
	if tipo == 'SMAX':
		nonzero = np.nonzero(yy_real)
		clase_real = S2 * yy_real

		clase_real = clase_real[nonzero]
		Smax = S2.max(axis=1)
		Smax = Smax[:, np.newaxis]
		S2_p = S2 - Smax
		exp = np.exp(S2_p)
		sum_exp = np.sum(exp, axis=1)
		loss = np.mean(-clase_real + np.log(sum_exp))
		# loss = np.mean(-np.log(np.exp(clase_real).T/sum_exp[:, np.newaxis]))

		inversa = 1 / sum_exp
		grad = inversa[:, np.newaxis] * exp
		grad[nonzero] -= 1

		return grad, loss


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
	y = acc_sum / y_est.shape[0]
	return y


def grad_sigmoide(x):
	y = sigmoide(x) * (1 - sigmoide(x))
	return y


def relu(x):
	# return np.maximum(0, x)
	# ceros = np.zeros((x.shape[0], x.shape[1]))
	return np.maximum(0, x)


def grad_relu(x):
	return np.where(x < 0, 0, 1)


def f_act(act, Y):
	if act == 'RELU':
		return relu(Y)
	elif act == 'SIG':
		return sigmoide(Y)
	elif act == 'LIN':
		return Y


def grad_fact(act, Y):
	if act == 'RELU':
		return grad_relu(Y)
	elif act == 'SIG':
		return grad_sigmoide(Y)
	elif act == 'LIN':
		return np.ones((Y.shape[0], Y.shape[1]))


def fit(x_train, y_train, x_test, y_test, tipo, nclases, nintermedia, batch_size, n_epocas, learning_rate, reg_lambda,
        n_train_data, act1, act2):
	im_shape = x_train.shape[1:]
	xtr = np.reshape(x_train[:n_train_data], (x_train[:n_train_data].shape[0], np.prod(im_shape)))
	x_mean = np.mean(xtr[:, np.newaxis], axis=0)

	xtr = (xtr) / 255

	xtr = np.hstack([np.ones((x_train[:n_train_data].shape[0], 1)), xtr])
	ytr = y_train[:n_train_data]

	yy_tr = np.zeros((ytr.shape[0], nclases))  # esto es un vector de train_data x 10
	yy_tr[np.arange(ytr.shape[0]), ytr.T] = 1  # vector de train_data x 10

	nro_batchs = int(ytr.shape[0] / batch_size)

	im_shape_test = x_test.shape[1:]
	xt = np.reshape(x_test, (x_test.shape[0], np.prod(im_shape_test)))
	xt = (xt) / 255
	xt = np.hstack([np.ones((x_test.shape[0], 1)), xt])
	yt = y_test
	yyy_t = np.zeros((yt.shape[0], nclases))
	yyy_t[np.arange(yt.shape[0]), yt.T] = 1

	W1 = np.random.randn(xtr.shape[1], nintermedia) * 0.01  # 3073 x 100, meto el primer bias
	W2 = np.random.randn(W1.shape[1] + 1, nclases) * 0.01  # 101 x 10 : meto el segundo bias
	train_acc = np.zeros((n_epocas, 1))
	train_loss = np.zeros((n_epocas, 1))
	test_loss = np.zeros((n_epocas, 1))
	test_acc = np.zeros((n_epocas, 1))
	epoca = []
	for i in range(n_epocas):
		indice = xtr.shape[0]
		indice = np.arange(indice)
		np.random.shuffle(indice)
		for j in range(nro_batchs):
			# Forward
			x_batch = xtr[indice[(j * batch_size):((j + 1) * batch_size)], :]  # de batch_size x 3073
			y_batch = ytr[indice[(j * batch_size):((j + 1) * batch_size)]]  # de batch_size x 10

			yyy_tr = yy_tr[indice[(j * batch_size):((j + 1) * batch_size)], :]

			# Capa 1:
			Y1 = np.dot(x_batch, W1)  # esto es un vector de batch_size x 101
			S1 = f_act(act1, Y1)
			S1_prima = np.hstack([np.ones((x_batch.shape[0], 1)), S1])  # esto es un vector de batch_size x 101
			# Capa 2
			S2 = np.dot(S1_prima, W2)  # es un vector de batch_size x 10
			S2_act = f_act(act2, S2)

			# Hago lo mismo para los datos de Test
			y_test_estimado = np.dot(xt, W1)
			y_test_estimado_S1 = f_act(act1, y_test_estimado)
			y_test_estimado_S1_prima = np.hstack([np.ones((xt.shape[0], 1)), y_test_estimado_S1])
			y_test_estimado_S2 = np.dot(y_test_estimado_S1_prima, W2)
			y_test_estimado_S2_act = f_act(act2, y_test_estimado_S2)

			# Calculo la regularización
			reg1 = np.sum(W1 * W1)
			reg2 = np.sum(W2 * W2)
			reg = reg1 + reg2
			# Cálculo de la Loss

			# 	Backward
			grad, loss_val = loss(S2_act, yyy_tr, tipo=tipo)
			grad = grad_fact(act2, S2) * grad

			grad_W2 = np.dot(S1_prima.T, grad)
			grad = np.dot(grad, W2.T)
			grad = grad[:, 1:]
			grad_f = grad_fact(act1, Y1)
			grad = grad_f * grad
			grad_W1 = np.dot(x_batch.T, grad)

			W1 = W1 - learning_rate * (grad_W1 + reg * W1)
			W2 = W2 - learning_rate * (grad_W2 + reg * W2)

			train_loss[i] += loss_val + (reg1 + reg2) * reg_lambda
			train_acc[i] += accuracy(S2_act, y_batch)
			a, tloss = loss(y_test_estimado_S2_act, yyy_t, tipo=tipo)
			test_loss[i] += tloss
			test_acc[i] += accuracy(y_test_estimado_S2_act, yt)

		epoca.append(i)
		train_acc[i] = train_acc[i] / nro_batchs
		train_loss[i] = train_loss[i] / nro_batchs
		test_acc[i] = test_acc[i] / nro_batchs

		print('Epoca: ' + str(i) + '  Accuracy: ' + str(train_acc[i]) + 'Loss: ' + str(train_loss[i]))
	print_resultados(epoca, test_loss, train_acc, test_acc, batch_size, n_train_data, n_epocas, train_loss, act1, act2)


def print_resultados(epoca, test_loss, train_acc, test_acc, batch_size, n_train_data, n_epocas, train_loss, act1, act2):
	plt.figure()
	plt.plot(epoca, train_acc, 'r*')
	plt.plot(epoca, test_acc, 'k*')
	plt.legend(['Training data', 'Testing data'])
	plt.title('Evolución del accuracy - CIFAR10')
	plt.xlabel('Época')
	plt.ylabel('Accuracy')
	plt.ylim(0, 0.5)
	plt.savefig('Accuracy{}bz{}td{} - {}{}.pdf'.format(n_epocas, batch_size, n_train_data, act1, act2))
	plt.figure()
	plt.plot(epoca, train_loss, 'r-*')
	plt.plot(epoca, test_loss, 'k-*')
	plt.title('Evolución de la Loss - CIFAR10')
	plt.xlabel('Época')
	plt.ylabel('Loss')
	plt.savefig('Loss_ep{}bz{}td{} - {} - {}{}.pdf'.format(n_epocas, batch_size, n_train_data, tipo, act1, act2))
	# plt.ylim(0,train_loss[1])
	plt.show()


loss_tipo = ['MSE', 'SMAX']
activacion = ['RELU', 'SIG', 'LIN']
act1 = activacion[0]
act2 = activacion[2]

tipo = loss_tipo[1]

nclases = 10  # salida
nintermedia = 100  # intermedia
batch_size = 256
n_epocas = 50
learning_rate = 1e-3
reg_lambda = 1
n_train_data = 20000

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

fit(x_train, y_train, x_test[:100], y_test[:100], tipo, nclases, nintermedia, batch_size, n_epocas, learning_rate,
    reg_lambda, n_train_data, act1, act2)
