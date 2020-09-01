import numpy as np
import matplotlib.pyplot as plt
from TP1_3KNN import KNN


class Datos:
	def __init__(self, nclases=2, ndatos=10, train=False, size=10):
		self.nclases = list(range(nclases))
		self.size = size
		self.train = train
		self.ndatos = ndatos
		self.datos = np.zeros((self.ndatos, 2))
		if train:
			datosx, datosy, self.clase = self.generar_datos()
		else:
			datosx, datosy = self.generar_datos()
		self.datos[:, 0] = datosx
		self.datos[:, 1] = datosy

	def generar_datos(self):  # genero datos bidimensionales
		x_data = np.zeros(self.ndatos)
		y_data = np.zeros(self.ndatos)
		if self.train:
			clase = []
		# for i in range(self.ndatos):
		x_data = np.random.random(self.ndatos) * self.size
		y_data = np.random.random(self.ndatos) * self.size
		if self.train:
			for i in range(self.ndatos):
				clase.append(np.random.choice(self.nclases))
		if self.train:
			return x_data, y_data, clase
		else:
			return x_data, y_data

	def print(self, clase=None, train=False):
		color = ['red', 'black', 'blue', 'orange', 'magenta', 'yellow']
		if train:
			for i in range(self.ndatos):
				plt.scatter(self.datos[i, 0], self.datos[i, 1], c=color[clase[i]], s=150)
		else:
			for i in range(self.ndatos):
				plt.scatter(self.datos[i, 0], self.datos[i, 1], c=color[clase[i]], edgecolors='black')
		plt.xlim(0,self.size)
		plt.ylim(0, self.size)

		plt.show(block=0)
		plt.pause(0.5)


size = 100
datos_train = Datos(train=True, ndatos=100, size=size, nclases=5)
# datos_train.print()
datos_test = Datos(ndatos=100, size=size)
datos_train.print(clase=datos_train.clase, train=True)
model = KNN(7)
model.train(datos_train.datos, datos_train.clase)

clase = model.predict(datos_test.datos)
datos_test.print(clase=clase)

xx, yy = np.meshgrid(np.arange(0, size, 5), np.arange(0, size, 5))
datos2 = np.c_[xx.ravel(), yy.ravel()]
clase2 = model.predict(datos2)
color = ['red', 'black', 'blue', 'orange', 'magenta', 'yellow']
for i in range(len(clase2)):
	plt.scatter(datos2[i, 0], datos2[i, 1], c=color[clase2[i]], alpha=0.8)
	plt.show(block=0)
	plt.pause(0.001)

plt.figure()


datos_train.print(clase=datos_train.clase, train=True)
model = KNN(5)
model.train(datos_train.datos, datos_train.clase)

clase = model.predict(datos_test.datos)
datos_test.print(clase=clase)

xx, yy = np.meshgrid(np.arange(0, size, 5), np.arange(0, size, 5))
datos2 = np.c_[xx.ravel(), yy.ravel()]
clase2 = model.predict(datos2)
color = ['red', 'black', 'blue', 'orange', 'magenta', 'yellow']
for i in range(len(clase2)):
	plt.scatter(datos2[i, 0], datos2[i, 1], c=color[clase2[i]], alpha=0.8)
	plt.show(block=0)
	plt.pause(0.001)

plt.figure()

datos_train.print(clase=datos_train.clase, train=True)
model = KNN(3)
model.train(datos_train.datos, datos_train.clase)

clase = model.predict(datos_test.datos)
datos_test.print(clase=clase)

xx, yy = np.meshgrid(np.arange(0, size, 5), np.arange(0, size, 5))
datos2 = np.c_[xx.ravel(), yy.ravel()]
clase2 = model.predict(datos2)
color = ['red', 'black', 'blue', 'orange', 'magenta', 'yellow']
for i in range(len(clase2)):
	plt.scatter(datos2[i, 0], datos2[i, 1], c=color[clase2[i]], alpha=0.8)
	plt.show(block=0)
	plt.pause(0.001)


plt.figure()


datos_train.print(clase=datos_train.clase, train=True)
model = KNN(1)
model.train(datos_train.datos, datos_train.clase)

clase = model.predict(datos_test.datos)
datos_test.print(clase=clase)

xx, yy = np.meshgrid(np.arange(0, size, 5), np.arange(0, size, 5))
datos2 = np.c_[xx.ravel(), yy.ravel()]
clase2 = model.predict(datos2)
color = ['red', 'black', 'blue', 'orange', 'magenta', 'yellow']
for i in range(len(clase2)):
	plt.scatter(datos2[i, 0], datos2[i, 1], c=color[clase2[i]], alpha=0.8)
	plt.show(block=0)
	plt.pause(0.001)