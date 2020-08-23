# Con Scikit-learn:








# ---------------------------------------------------------------------------------
# Sin usar Scikit-learn:


import numpy as np
import matplotlib.pyplot as plt
import time



class Lineal:  # Tomado del TP0_5.py
	def __init__(self, a=0, b=0):
		self.a = a
		self.b = b

	def __call__(self, x):
		return self.a * x + self.b


class Regresion(object):
	def __init__(self, n=1, cant_datos=10, size=40, tipo='Lineal'):
		self.n = n  # Dimensión del problema
		self.type = tipo
		self.x_data = None
		self.y_data = None
		self._size = size
		self.coef = None
		self.x = list(range(size + 1))
		self.g = None
		self.cant_datos = cant_datos
		self.generar_datos()


	def generar_datos(self):
		# cant_datos = 100
		x = np.random.rand(self.cant_datos, self.n) * self._size
		self.x_data = np.concatenate((np.ones((self.cant_datos, 1)), x), axis=1)
		self.y_data = np.random.rand(self.cant_datos) * self._size * 2
		self.y_data = np.reshape(self.y_data, (self.cant_datos, 1))

	# print(self.x_data)
	# print(self.y_data)

	def linear(self):
		self.coef = np.matmul(
			np.matmul(np.linalg.inv(np.matmul(np.transpose(self.x_data), self.x_data)), np.transpose(self.x_data)),
			self.y_data)
		L = Lineal(self.coef[1], self.coef[0])
		self.g = np.vectorize(L.__call__)
		return self.coef

	def print(self):
		plt.scatter(self.x_data[:, 1], self.y_data)
		plt.plot(self.x, self.g(self.x), color='red')


start = []
end = []
start2 = []
end2 = []
c_datos = []
for i in range(1,1000):
	start.append(time.time())
	c_datos.append(i*10)
	R = Regresion(n=1, cant_datos=i*10)
	coef = R.linear()
	end.append(time.time())
	start2.append(time.time())
	R = Regresion(n=2, cant_datos=i*10)
	coef = R.linear()
	end2.append(time.time())
	# print(i)
plt.figure(2)
dif = np.subtract(end, start)
dif2 = np.subtract(end2, start2)
plt.plot(c_datos, dif)
plt.plot(c_datos, dif2)
# R.print()

plt.show()

