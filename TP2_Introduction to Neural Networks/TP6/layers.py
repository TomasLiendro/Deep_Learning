import numpy as np


class BaseLayer:
	def __init__(self):
		pass

	def get_output_shape(self):
		pass

	def set_output_shape(self, *args):
		pass


# class ConcatInput(BaseLayer):
# 	def __init__(self, x1, x2):
# 		super().__init__()
#
# 	def get_output_shape(self):
# 		return self.c_dim
#
# 	def set_output_shape(self):
#
# 	def __call__(self):


class WLayer(BaseLayer):
	def __init__(self, units=None, input_dim=None):
		super().__init__()
		self.type = None
		self.x_dim = input_dim
		self.units = units

	def get_weight(self):
		return self.W
	#
	# def update_weights(self):


class Dense(WLayer):
	def __init__(self, units=None, activation=None, input_dim=None):
		super().__init__(units=units, input_dim=input_dim)
		self.type = 'Dense'
		self.W = None
		self.activation = activation

	def __call__(self, x):
		x_prima, prod = self.dot(x)
		S = self.activation(prod)
		return x_prima, prod, S

	def init_weights(self, units=1, x_dim=1):
		self.W = np.random.randn(x_dim + 1, units) * 1e-2

	def dot(self, x):
		x_prima = np.hstack([np.ones((x.shape[0], 1)), x])
		return x_prima, np.dot(x_prima, self.W)


class Concatenate(BaseLayer):
	def __init__(self, input_dim=None):
		super().__init__()
		self.type = 'Concatenate'
		self.input_dim = input_dim
		self.units = None

	def set_output_shape(self, shape_previous):
		self.units = shape_previous + self.input_dim

	def __call__(self, x1, x2):
		S_concatenate = np.concatenate([x1, x2], axis=1)
		return x2, S_concatenate, S_concatenate
