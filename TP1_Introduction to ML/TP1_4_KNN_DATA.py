import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from TP1_3KNN import KNN


class Datos:
	def __init__(self, nclases=5, ndatos=50, size=100):
		self.nro_clases = nclases
		self.nclases = list(range(nclases))
		self.size = size
		self.ndatos = ndatos
		self.datos = np.zeros((self.ndatos, 2))

		self.datosx, self.datosy, self.clase = self.generar_datos()
		self.datos[:, 0] = self.datosx
		self.datos[:, 1] = self.datosy
		self.rang = np.arange(self.size / len(self.nclases), self.size + self.size / len(self.nclases),
		                      self.size / len(self.nclases))

	def generar_datos(self):  # genero datos bidimensionales
		x_data = np.zeros(self.ndatos)
		y_data = np.zeros(self.ndatos)
		clase = []
		rangos = np.arange(self.size / len(self.nclases), self.size + self.size / len(self.nclases),
		                   self.size / len(self.nclases))
		self.rang = rangos
		x_data = np.random.normal(np.random.uniform(-self.size / 2, self.size / 2, 1),
		                          np.random.uniform(self.size / 5, int(self.size)), self.ndatos)
		y_data = np.random.normal(np.random.uniform(-self.size / 2, self.size / 2, 1),
		                          np.random.uniform(self.size / 5, int(self.size)), self.ndatos)
		dist = []
		for i in range(self.ndatos):
			dist.append(np.linalg.norm([x_data[i], y_data[i]], ord=2))
			for j in range(self.nro_clases):

				if dist[i] > rangos[-1 - j]:
					clase.append(self.nro_clases - j)
					break
			if dist[i] < rangos[0]:
				clase.append(0)

		return x_data, y_data, clase

	def print(self, clase=None, train=False):
		cmap_bold = ListedColormap(['#0037FF', '#FF8800', '#007F46', '#FF0D05', '#FF7E05'])

		if train:
			plt.scatter(self.datosx, self.datosy, c=clase, cmap=cmap_bold)
		else:
			plt.scatter(self.datosx, self.datosy, c=clase)
		plt.xlim(-1.5 * self.size, 1.5 * self.size)
		plt.ylim(-1.5 * self.size, 1.5 * self.size)


size = 100
ndatos = 1000
datos_train = Datos(ndatos=ndatos, size=size)
datos_test = Datos(ndatos=50, size=size)
cmap_light = ListedColormap(['#00a5ff', '#FFB45E', '#00E016', '#FF8987', '#FF9927'])

for k in [1, 3, 5, 7]:
	print("K: {}".format(k))
	plt.figure(k)

	model = KNN(k)
	model.train(datos_train.datos, datos_train.clase)
	clase = model.predict(datos_test.datos)
	accuracy = model.acc(datos_test.clase, clase)
	print(accuracy)

	xx, yy = np.meshgrid(np.arange(-2 * size, 2 * size, 2), np.arange(-2 * size, 2 * size, 2))
	datos2 = np.c_[xx.ravel(), yy.ravel()]
	clase2 = model.predict(datos2)
	Z = clase2.reshape(xx.shape)

	plt.pcolormesh(xx, yy, Z, cmap=cmap_light, shading='auto')
	datos_train.print(train=True, clase=datos_train.clase)
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('K={}, acc={}, ndatos={}'.format(k, '%.3f' % accuracy, ndatos))
	plt.show(block=0)
	plt.pause(0.001)

	# plt.savefig("fronteras_de_decision_K_{}_{}.pdf".format(k, ndatos))

datos_train.print(clase=datos_train.clase, train=True)
