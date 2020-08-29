import numpy as np
import matplotlib.pyplot as plt


class Lineal:  # Tomado del TP0_5.py
	def __init__(self, a):
		self.a = a
		# self.b = b

	def __call__(self, x):
		return self.a * x


class Regresion(object):
	def __init__(self, n=1, cant_datos=10, size=40, tipo='Lineal'):
		self.n = n  # Dimensión del problema
		self.type = tipo
		self.x_data = None
		self.y_data = None
		self._size = size
		self.coef = None
		self.x2 = None
		self.x = list(range(size + 1))
		self.g = None
		self.cant_datos = cant_datos
		self.generar_datos()

	def generar_datos(self):
		ordenada = 2
		slope = 3
		self.x2 = np.random.rand(self.cant_datos, self.n) * self._size
		self.x_data = np.concatenate((np.ones((self.cant_datos, 1)), self.x2), axis=1)
		pendiente = np.ones((self.n, 1)) * slope
		self.y_data = np.matmul(self.x2, pendiente) + 5 * np.random.uniform(-1, 1, (self.cant_datos, 1)) + ordenada
		self.y_data = np.reshape(self.y_data, (self.cant_datos, 1))

	def entrenar(self, ndatos):  # tomo los primeros n datos para entrenar
		self.coef = np.matmul(
			np.matmul(np.linalg.inv(np.matmul(np.transpose(self.x_data[0:ndatos - 1]), self.x_data[0:ndatos - 1])), np.transpose(self.x_data[0:ndatos - 1])), self.y_data[0:ndatos - 1])

	def calc_recta(self, x, a):
		# L = Lineal(self.coef[1], self.coef[0])
		L = Lineal(a)
		self.g = np.vectorize(L.__call__)
		return self.g(x)

	def verificar(self, cant_datos):
		# tiene que calcular el error:
		error = 0
		for j in range(len(self.y_data)):
			Y_calc = self.calc_recta(self.x_data[j], self.coef)
			error += (self.y_data[j] - Y_calc[j])**2
		return error/cant_datos

	def print(self):
		plt.scatter(self.x_data[:, 1], self.y_data)
		plt.plot(self.x2, self.g(self.x2), color='red')


cant_datos = 100
nverif = cant_datos
error = []
dim = []
for i in range(1, 20):
	regresion = Regresion(n=i, cant_datos=cant_datos)
	regresion.entrenar(nverif)
	error.append(regresion.verificar(cant_datos))
	dim.append(i)

plt.plot(dim, error, 'k*')
plt.show()
