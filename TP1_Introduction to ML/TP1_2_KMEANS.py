import matplotlib.pyplot as plt
import numpy as np


class Clase(object):
	def __init__(self, ndim=3, nclases=4, n_distr=2):
		self.ndistr = n_distr
		self.dimension = ndim
		self.K = nclases

		self.clase = []  # hay K clases
		self.media = []  # hay K clases por lo que hay K medias
		self.nueva_clase = []
		self.datos = []
		self.tamano = 100

		self.color = []

		for i in range(self.K):
			self.color.append('#%06X' % np.random.randint(0, 0xFFFFFF))

		for i in range(self.ndistr):
			self.start()

		self.datos2 = []
		for d in range(self.ndistr):
			for i in range(len(self.datos[d])):
				self.datos2.append([self.datos[d][i, :], 0])  # el ultimo indice indica la clase a la que pertenece

		self.definir_medias()

		if self.dimension == 2:
			for l in range(len(self.datos2)):
				plt.plot(self.datos2[l][0][0], self.datos2[l][0][1], 'k*')
			for k in range(self.K):
				plt.plot(self.media[k][0][0], self.media[k][0][1], 'r+')

		plt.show(block=0)
		plt.pause(1)
		plt.clf()

		# plt.
		self.iterar()

	def start(self):
		mean = np.random.uniform(-1, 1) * self.tamano
		std = np.random.random() * 200
		ndatos = np.random.randint(1, 1000)
		self.datos.append((np.random.normal(mean, std, (ndatos, self.dimension))))

	def definir_medias(self):
		for i in range(self.K):
			self.media.append(np.random.uniform(-self.tamano, self.tamano, (1, self.dimension)))

	def calc_media(self):
		self.media = []
		for k in range(self.K):
			sum = np.zeros((1, self.dimension))
			cant = 0
			for i in range(len(self.datos2)):

				if self.datos2[i][1] == k:
					sum = self.datos2[i][0] + sum
					cant += 1
			self.media.append(sum / cant)

	def iterar(self):
		niter = 20
		for h in range(niter):
			self.kmeans()
			self.calc_media()

			# self.print()

			if h == 0:
				plt.figure()
				self.print()
				plt.savefig('ej2_t0.pdf')
			if h == 5:
				plt.figure()
				self.print()
				plt.savefig('ej2_ti.pdf')
			if h == 10:
				plt.figure()
				self.print()
				plt.savefig('ej2_ti2.pdf')
			if h == 19:
				plt.figure()
				self.print()
				plt.savefig('ej2_te.pdf')

	def kmeans(self):
		for j in range(len(self.datos2)):  # calculo la distancia para cada elemento de la clase a la media
			# de la p-ésima clase
			dist = []
			if j >= len(self.datos2):
				break
			for p in range(self.K):
				dist.append(np.linalg.norm(self.media[p] - self.datos2[j][0], ord=2))
			indice = dist.index(min(dist))
			self.datos2[j][1] = indice

	def print(self):
		for l in range(len(self.datos2)):
			plt.plot(self.datos2[l][0][0], self.datos2[l][0][1], '*', color=self.color[self.datos2[l][1]])
		plt.title("K-Means")
		plt.xlabel('X')
		plt.ylabel('Y')

		plt.show(block=0)
		plt.pause(0.01)


nclases = 4
n_distr = 2
clase = Clase(ndim=2, nclases=nclases, n_distr=n_distr)
