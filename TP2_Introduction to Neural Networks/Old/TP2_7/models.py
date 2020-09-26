import numpy as np
import Old.TP2_7.losses as losses
import Old.TP2_7.optimizers as optimizers
import Old.TP2_7.metrics as metrics
import Old.TP2_7.regularizers as reg

import matplotlib.pyplot as plt

class Network:
	def __init__(self):
		self.layers = []
		self.S = []
		self.acc_train = None
		self.loss_train = None
		self.acc_test = None
		self.loss_test = None
		self.epoca = None
		self.epoca_arr = []

		self.loss = None
		self.opt = None
		self.reg = None
		self.problem_name = 'XOR'
		self.tdata = False

	def add(self, layer):
		type_layer = layer.type
		self.layers.append((type_layer, layer))
		if self.get_layer(0)[0] == 'Dense':
			if len(self.layers) == 1:
				layer.init_weights(units=layer.units, x_dim=layer.x_dim)
			else:
				layer.init_weights(units=layer.units, x_dim=self.layers[-2][1].units)
			print('Dense layer created.')

		elif self.get_layer(0)[0] == 'Concatenate':
			if len(self.layers) == 1:
				return Exception, 'First layer cannot be of type: \'Concatenate\''
			else:
				layer.set_output_shape(self.layers[-2][1].units)
			print('Concatenate layer created.')
		else:
			return Exception, 'First layer not recognize, please set first layer as type: \'Dense\''

	def get_layer(self, n_of_layer):
		return self.layers[n_of_layer - 1]

	def fit(self, x, y, test_data=None, epochs=100, loss=losses.MSE(), opt=optimizers.BGD(lr=None, bs=None), name='XOR', reg=reg.L2(lam=0)):
		self.acc_train = np.zeros(epochs)
		self.loss_train = np.zeros(epochs)
		self.acc_test = np.zeros(epochs)
		self.loss_test = np.zeros(epochs)
		self.loss = loss
		self.opt = opt
		self.reg = reg
		self.problem_name = name
		self.tdata = True if test_data is not None else False

		im_shape = x.shape[1:]
		x = np.reshape(x, (x.shape[0], np.prod(im_shape)))
		x_mean = np.mean(x[:, np.newaxis], axis=0)
		# x = (x - x_mean)
		if self.tdata:
			xtest = np.reshape(test_data[0], (test_data[0].shape[0], np.prod(im_shape)))

		print('Starting training...')
		for e in range(epochs):
			self.epoca = e
			self.epoca_arr.append(e)
			self.opt(x, y, self)
			print('Epoca: ' + str(e) + 'Acc: ' + str(self.acc_train[e]))

			if self.tdata:
				self.predict(xtest)
				ytest = self.S[-1][2]
				self.acc_test[self.epoca] += metrics.Accuracy.__call__(metrics.Accuracy(), y_score=ytest, y_real=test_data[1],
				                                                       problem_name=self.problem_name)
				reg_sum = 0
				if self.reg is not None:
					for i in range(len(self.layers)):
						if self.layers[i][0] == 'Dense':
							reg_sum += self.reg(W=self.layers[i][1].W)
				self.loss_test[self.epoca] += self.loss(ytest, test_data[1]) + reg_sum
				self.S = []

		self.print()

	def forward_upto(self, j, x):
		x_input = x
		for k in range(1, j+1):
			if k == 1:
				if self.get_layer(k)[0] == 'Dense':
					self.S.append(self.get_layer(1)[1].__call__(x_input))
				elif self.get_layer(k)[0] == 'Concatenate':
					return Exception, 'First layer cannot be of type: \'Concatenate\''
				else:
					return Exception, 'First layer not recognized, please set first layer as type: \'Dense\''
			else:
				if self.get_layer(k)[0] == 'Dense':
					self.S.append(self.get_layer(k)[1].__call__(self.S[-1][2]))
				elif self.get_layer(k)[0] == 'Concatenate':
					self.S.append(self.get_layer(k)[1].__call__(x_input, self.S[-1][2]))

	def predict(self, x):  ######
		self.forward_upto(len(self.layers), x)
		# return np.where(np.argmax(self.S[-1]) == 0, -1, 1)

	def backward(self, x, y):
		self.predict(x)
		y_est = self.S[-1][2]
		self.acc_train[self.epoca] += metrics.Accuracy.__call__(metrics.Accuracy(), y_score=y_est, y_real=y, problem_name=self.problem_name)
		reg_sum = 0
		if self.reg is not None:
			for i in range(len(self.layers)):
				if self.layers[i][0] == 'Dense':
					reg_sum += self.reg(W=self.layers[i][1].W)
		self.loss_train[self.epoca] += self.loss(y_est, y) + reg_sum
		grad = self.loss.gradient(y_est, y)
		dW = 0
		for i in range(len(self.layers)):
			if self.get_layer(-i)[0] == 'Dense':
				grad *= self.get_layer(-i)[1].activation.gradient(self.S[-i-1][1])
				dW = self.dot(self.S[-i-1][0].T, grad)
				grad = self.dot(grad, self.get_layer(-i)[1].W.T)
				grad = grad[:, 1:]
				self.get_layer(-i)[1].W = self.opt.update_W(self.get_layer(-i)[1].W, dW)

			elif self.get_layer(-i)[0] == 'Concatenate':
				grad = grad[:, self.get_layer(-i)[1].input_dim:]

		self.S = []

	def dot(self, x1, x2):
		return np.dot(x1, x2)

	def print(self):
		plt.figure(1)
		plt.plot(self.epoca_arr, self.acc_train, '-')
		if self.tdata:
			plt.plot(self.epoca_arr, self.acc_test, 'k*-')
		plt.legend(['Training data', 'Testing data'])
		plt.title('Evolución del accuracy - XOR')
		plt.xlabel('Época')
		plt.ylabel('Accuracy')
		plt.ylim(0, 102)
		# plt.savefig('TP2_7.pdf')
		plt.figure(2)
		plt.plot(self.epoca_arr, self.loss_train, '-')
		if self.tdata:
			plt.plot(self.epoca_arr, self.loss_test, 'k*-')
		plt.legend(['Training data', 'Testing data'])
		plt.title('Evolución de la Loss - XOR')
		plt.xlabel('Época')
		plt.ylabel('Loss')
		# plt.savefig('Loss_TP2_7.pdf')
		# plt.show()
