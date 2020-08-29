import matplotlib.pyplot as plt
import numpy as np


class Regresion(object):
	def __init__(self, n=1, cant_datos=10, size=40):
		self.n = n  # Dimensión del problema
		self._size = size
		self.cant_datos = cant_datos

		self.x = None
		self.x_data = None
		self.y_data = None
		self.coef = None

		self.generar_datos()

	def generar_datos(self):
		ordenada = 2
		slope = 3
		pendiente = np.ones((self.n, 1)) * slope

		self.x = np.random.rand(self.cant_datos, self.n) * self._size
		self.x_data = np.concatenate((np.ones((self.cant_datos, 1)), self.x), axis=1)

		self.y_data = np.matmul(self.x, pendiente) + 1 * np.random.uniform(-1, 1, (self.cant_datos, 1)) + ordenada
		self.y_data = np.reshape(self.y_data, (self.cant_datos, 1))

	def calcular_coef(self):
		self.coef = np.matmul(
			np.matmul(np.linalg.inv(np.matmul(np.transpose(self.x_data), self.x_data)), np.transpose(self.x_data)),
			self.y_data)

	def calcular_error(self):
		error = 0
		y_calc = self.calc_recta()
		# print(y_calc)
		# plt.plot(self.x, self.y_data, '*')
		# plt.plot(self.x, y_calc, '*')

		plt.show()
		# for j in range(len(self.y_data)):
		# print(y_calc[j], self.y_data[j])
		error = sum((y_calc - self.y_data) ** 2)
		print(error)
		return error / self.cant_datos

	def calc_recta(self):
		return np.matmul(self.x_data, self.coef)

	def print(self):
		plt.scatter(self.x, self.y_data)
		plt.plot(self.x, np.matmul(self.x_data, self.coef), color='red')


cant_datos = 100
error = []
dim = []
for i in range(1, 50):
	regresion = Regresion(n=i, cant_datos=cant_datos)
	regresion.calcular_coef()
	error.append(regresion.calcular_error())
	dim.append(i)
# regresion.print()
# plt.show()

plt.plot(dim, error, 'k*')
plt.show()
