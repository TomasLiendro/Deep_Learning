import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mrk


class R2:
	def __init__(self, x=None, y=None):
		self.x = x
		self.y = y


class Pez:
	def __init__(self,x=None,y=None):
##		super().__init__()
		self.pos = R2(x,y)
		self.vel = R2(x,y)
		self.maxVelPez = 15


class Cardumen:
	def __init__(self):  # , size, maxVel, maxDist):
##		super().__init__()
		self._size = None
		self._maxVel = None
		self._maxDist = None
		self._dt = 0.1  # segundos

		self.npeces = 16
		self.pez = []
		self.centro_cardumen = Pez()

	def initialize(self, size, maxVel, maxDist):
		self._size = size
		self._maxVel = maxVel
		self._maxDist = maxDist
		for i in range(self.npeces):
			self.pez.append(Pez())
			# Define una posición aleatoria de los peces
			self.pez[i].pos.x = np.random.uniform(0, 1) * self._size
			self.pez[i].pos.y = np.random.uniform(0, 1) * self._size
			# Define una velocidad aleatoria de los peces
			self.pez[i].vel.x = np.random.uniform(-1, 1) * self._maxVel
			self.pez[i].vel.y = np.random.uniform(-1, 1) * self._maxVel
			# Verifica que el módulo de la velocidad no supere la velocidad máxima
			while np.linalg.norm([self.pez[i].vel.x, self.pez[i].vel.y], ord=2) > self._maxVel:
				self.pez[i].vel.x = np.random.uniform(-1, 1) * self._maxVel
				self.pez[i].vel.y = np.random.uniform(-1, 1) * self._maxVel

	def calcular_centro(self):
		sum = Pez(0, 0)
		# Calcula posición y velocidad media del cardumen
		for i in range(self.npeces):
			sum.pos.x += self.pez[i].pos.x
			sum.pos.y += self.pez[i].pos.y
			sum.vel.x += self.pez[i].vel.x
			sum.vel.y += self.pez[i].vel.y
		self.centro_cardumen.pos.x = sum.pos.x / self.npeces
		self.centro_cardumen.pos.y = sum.pos.y / self.npeces
		self.centro_cardumen.vel.x = sum.vel.x / self.npeces
		self.centro_cardumen.vel.y = sum.vel.y / self.npeces

	def dv1_fun(self, index_pez):
		dv1 = R2(0, 0)
		dv1.x, dv1.y = (self.centro_cardumen.pos.x - self.pez[index_pez].pos.x)/8, (self.centro_cardumen.pos.y - self.pez[index_pez].pos.y)/8
		return dv1

	def dv2_fun(self, index_pez):
		dv2 = R2(0, 0)
		for i in range(self.npeces):
			if i is not index_pez:
				dist = np.linalg.norm([self.pez[i].pos.x - self.pez[index_pez].pos.x, self.pez[i].pos.y - self.pez[index_pez].pos.y], ord=2)
				if dist < self._maxDist:
					dv2.x += (-self.pez[i].pos.x + self.pez[index_pez].pos.x)/dist
					dv2.y += (-self.pez[i].pos.y + self.pez[index_pez].pos.y)/dist
		return dv2

	def dv3_fun(self, index_pez):
		dv3 = R2(0, 0)
		dv3.x, dv3.y = (self.centro_cardumen.vel.x - self.pez[index_pez].vel.x)/8, (self.centro_cardumen.vel.y - self.pez[index_pez].vel.y)/8
		return dv3

	def calc_deltaVel(self, index_pez):
		dv = R2(0, 0)
		dv1 = self.dv1_fun(index_pez)
		dv2 = self.dv2_fun(index_pez)
		dv3 = self.dv3_fun(index_pez)
		dv.x = dv1.x + dv2.x + dv3.x
		dv.y = dv1.y + dv2.y + dv3.y
		return dv

	def doStep(self):
		self.calcular_centro()
		for i in range(self.npeces):
			delta_vel = self.calc_deltaVel(i)
			# norm_vel = np.linalg.norm([self.pez[i].vel.x, self.pez[i].vel.y], ord=2)
			# norm_dvel = np.linalg.norm([delta_vel.x, delta_vel.y], ord=2)
			nueva_vel = np.linalg.norm([self.pez[i].vel.x + delta_vel.x, self.pez[i].vel.y + delta_vel.y], ord=2)
			# if (norm_vel+norm_dvel) > self.pez[i].maxVelPez:
			if nueva_vel < self.pez[i].maxVelPez:
				self.pez[i].vel.x += delta_vel.x
				self.pez[i].vel.y += delta_vel.y
			self.pez[i].pos.x += self.pez[i].vel.x * self._dt
			self.pez[i].pos.y += self.pez[i].vel.y * self._dt

	def print(self):
		for i in range(self.npeces):
			plt.plot(self.centro_cardumen.pos.x, self.centro_cardumen.pos.y, marker='+', color='black', markersize=8)
			# plt.figure(1)
			plt.plot(self.pez[i].pos.x, self.pez[i].pos.y, marker='o')
		plt.xlim(0, self._size)
		plt.ylim(0, self._size)
		# plt.show(block=False)
		# plt.pause(0.01)
		# plt.close()


c = Cardumen()
size = 40
maxVel = 5
maxDist = 1
niter = 1000
#
c.initialize(size, maxVel, maxDist)
for i in range(niter):
	c.doStep()
	if i == 0:
		plt.figure(1)
		c.print()
	if i == 10:
		plt.figure(2)
		c.print()
	if i == 20:
		plt.figure(3)
		c.print()
	if i == 30:
		plt.figure(4)
		c.print()
	if i == 40:
		plt.figure(5)
		c.print()

plt.show()
