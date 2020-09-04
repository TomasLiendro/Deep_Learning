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

		self.y_data = np.matmul(self.x, pendiente) + 5 * np.random.uniform(-1, 1, (self.cant_datos, 1)) + ordenada
		self.y_data = np.reshape(self.y_data, (self.cant_datos, 1))

	def calcular_coef(self):
		self.coef = np.matmul(
			np.matmul(np.linalg.inv(np.matmul(np.transpose(self.x_data), self.x_data)), np.transpose(self.x_data)),
			self.y_data)

	def ECM(self, xd, d):
		yc = self.coef[d] * xd
		yc = yc.reshape(self.cant_datos, 1)
		ECM1 = [(yc[j] - self.y_data[j]) ** 2 for j in range(len(yc))]
		ECMt = (1 / self.cant_datos) * sum(ECM1)
		return ECMt

	def calc_recta(self):
		return np.matmul(self.x_data, self.coef)

	def print(self):
		plt.scatter(self.x_data[:, 1], self.y_data[:, 0])
		plt.plot(self.x, np.matmul(self.x_data, self.coef), color='red')
		plt.xlabel('X')
		plt.ylabel('Y')
		plt.title('Conjunto de datos')
		plt.legend(['Regresión lineal', 'Datos originales'])
		plt.savefig('Datos_ej1.pdf')


cant_datos = 100
error1 = []
error2 = []

dim = []
ndatos = []
d = 1

for i in range(1, 100):
	regresion = Regresion(n=10, cant_datos=cant_datos * i)
	xd = np.linspace(0, 1, cant_datos * i)

	regresion.calcular_coef()
	error1.append(regresion.ECM(xd, d))
	ndatos.append(cant_datos * i)

xd = np.linspace(0, 1, cant_datos)
for i in range(1, 100):
	regresion = Regresion(n=i, cant_datos=cant_datos)
	regresion.calcular_coef()
	error2.append(regresion.ECM(xd, d))
	dim.append(i)

plt.figure()
plt.plot(ndatos, error1, 'k*')
plt.xlabel('Cantidad de datos')
plt.ylabel('Error')
plt.title('Error Cuadrático Medio')

plt.figure()
plt.plot(ndatos, error2, 'k*')
plt.xlabel('Dimensión')
plt.ylabel('Error')
plt.title('Error Cuadrático Medio')
plt.show()
# plt.savefig('Error2_ej1.pdf')
