import numpy as np


class Linear_classifier(object):
	def __init__(self):
		self.im_shape = None
		self.X = None
		self.Y = None
		self.W = None
		self.batch_size = None

	def fit(self, X, Y):
		self.im_shape = X.shape[1:]
		self.X = np.reshape(X, (np.prod(self.im_shape)), X.shape[0])
		self.X = np.concatenate((self.X, np.ones(X.shape[0])))
		self.Y = Y
		self.batch_size = 128
		self.W = np.random.randn(np.amax(self.Y), np.prod(self.im_shape) + 1)
		self.loss_gradient()

	def predict(self):
		assert self.X is not None, 'Train method needs to be call first'

	def loss_gradient(self, f):
		"""
		a naive implementation of numerical gradient of f at x
		- f should be a function that takes a single argument
		- x is the point (numpy array) to evaluate the gradient at
		"""
		X_batch = self.
		x = self.W
		fx = f(X_batch)  # evaluate function value at original point
		grad = np.zeros(x.shape)
		h = 0.00001

		# iterate over all indexes in x
		it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
		while not it.finished:
			# evaluate function at x+h
			ix = it.multi_index
			old_value = x[ix]
			x[ix] = old_value + h  # increment by h
			fxh = self(x)  # evalute f(x + h)
			x[ix] = old_value  # restore to previous value (very important!)

			# compute the partial derivative
			grad[ix] = (fxh - fx) / h  # the slope
			it.iternext()  # step to next dimension

		return grad


class SVM(Linear_classifier):

	def Loss(self, X):
		delta = 1.0
		scores = self.W.dot(X)
		loss = np.array([])
		reg = np.sum(np.square(self.W))
		lam = 1.1
		for y in range(np.amax(self.Y)):
			# compute the margins for all classes in one vector operation
			margins = np.maximum(0, scores - scores[y] + delta)
			# on y-th position scores[y] - scores[y] canceled and gave delta. We want
			# to ignore the y-th position and only consider margin on max wrong class
			margins[y] = 0
			loss = np.append(loss, [np.sum(margins)], 1)
		L = 1 / X.shape[0] * np.sum(loss) + lam * reg
		return L

# class SoftMax(Linear_classifier):
