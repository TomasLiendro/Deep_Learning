# Ejercicio 1
import numpy as np

A = np.array([[1, 0, 1],[2, -1, 1],[-3, 2, -2]])
b = np.array([[-2], [1], [-1]])
Ainv = np.linalg.inv(A)
x = np.matmul(Ainv, b)
print(x)

# Ejercicio 2
import numpy as np
import matplotlib.pyplot as plt

k = 3
th = 2
fig = plt.figure(1)
data = np.random.gamma(k, th, 1000)
hist = np.histogram(data)
mean = np.mean(data)
std = np.std(data)
plt.hist(data, 50)
plt.text(x=mean+0.1, y=70, s=r'Mean value: '+r'%.3f'%mean)
plt.plot([mean, mean],[0,70])
plt.title('Ejercicio 2: distribucion Gamma')
plt.xlabel('x')
plt.ylabel('Distribucion Gamma')
print('Resultados empíricos')
print(mean, std)
# Para verificar los resultados anteriores calculo los valores teóricos
mean_teorico = k*th
std_teorico = np.sqrt(k*th**2)
print('Resultados teóricos')
print(mean_teorico, std_teorico)
plt.savefig('Ej2.pdf')
# plt.show()

# Ejercicio 3
import numpy as np


def Bhaskara(a, b=0, c=0):
	arg = b ** 2 - 4 * a * c  # dentro de la raiz cuadrada
	if arg >= 0:  # si el argumento de la raiz es positivo
		x2 = (-b + np.sqrt(arg)) / (2 * a)  # sol 1
		x1 = (-b - np.sqrt(arg)) / (2 * a)  # sol 2
		im = np.array([[x1], [x2]])  # armo el array
	else:  # si el argumento de la raiz es positivo
		arg = -arg  # tomo el valor abs de arg
		x1 = complex(-b/(2*a), np.sqrt(arg) / (2 * a))  # sol 1
		x2 = complex(-b/(2*a), -np.sqrt(arg) / (2 * a))  # sol 2
		im = np.array([[x1], [x2]])  # armo el array
		print('Complex number! ')
	return im
# print(Bhaskara(22, 19, 7))

# Ejercicio 4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mkr
from TP0_3 import Bhaskara


def plot_parabola(a, b=0, c=0):
	step = 0.1
	x = np.linspace(-4.5, 4, 1000)
	y = np.zeros((len(x),1))
	index = 0
	for i in x:
		y[index] = a*i**2+b*i+c
		index += 1
	roots = Bhaskara(a, b, c)

	fig = plt.figure(1)
	plt.plot(x,y, marker=None)
	plt.title('Ejercicio 4: Parábola')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.plot(roots[0], 0, marker='o', color='red')
	plt.plot(roots[1], 0, marker='o', color='red')
	plt.text(roots[0]-1.5, -1, r'$x_1=$' + str('%.3f'%roots[0]))
	plt.text(roots[1], -1, r'$x_2=$' + str('%.3f'%roots[1]))
	plt.grid()
	plt.savefig('TP0_4.pdf')
	plt.show()
# plot_parabola(1, 1, -2)  # llamado a la funcion de plot

# Ejercicio 5


class Lineal:
	def __init__(self, a=0, b=0):
		self.a = a
		self.b = b

	def __call__(self, x):
		return self.a * x + self.b


'''
# Uncomment this: 
equation = Lineal(1, 2)
solution = equation(3)
print(Lineal.__call__)
print(solution)
'''

# Ejercicio 6
from TP0_5 import Lineal


class Exponential(Lineal):
	def __call__(self, x=0):
		return self.a * x ** self.b

'''
# Uncomment this:
equation = Exponential(1, 2)
solution = equation(3)
print(solution)
'''

# Ejercicio 7
import circunferencia as circle
from circunferencia import PI, area
from numpy import pi

if circle.PI is pi:
	solution1 = circle.area(10)
	print('%3.f' % solution1)
else:
	print(ValueError)

if PI is pi:
	solution2 = area(10)
	print('%3.f' % solution2)
else:
	print(ValueError)

print('They are the same object!' if solution1 is solution2 else 'They are different objects!')
print(id(solution1), id(solution2))

# Ejercicio 8

import geometria.rectangulo as rectangle
import geometria.circunferencia as circle
import geometria
# print(geometria.circunferencia.area(1))
# print(rectangle.area(1, 2))
# print(circle.area(1))

# Ejercicio 9
import p0_lib
from p0_lib import rectangulo
from p0_lib.circunferencia import PI, area
from p0_lib.elipse import area
from p0_lib.rectangulo import area as area_rect

# Uncomment this
'''
print(p0_lib.rectangulo.area(1, 2))
print(rectangulo.area(1, 2))
print(area(1,2))  # --> Es el área de la elipse. El area de circ. queda pisada
print(area_rect(1, 2))
'''

# Ejercicio 10
import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
	return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)


n = 10
x = np.linspace(-3, 3, 4 * n)
y = np.linspace(-3, 3, 3 * n)

X, Y = np.meshgrid(x, y)

plt.imshow(f(X, -Y), cmap='bone')
plt.xticks(())
plt.yticks(())

plt.colorbar(shrink=0.82)

plt.show()

# Ejercicio 11
import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
	return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)


n = 256
x = np.linspace(-3, 3, 4 * n)
y = np.linspace(-3, 3, 3 * n)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.xticks(())
plt.yticks(())
plt.contourf(Z, cmap='hot', alpha=0.8)
cnt = plt.contour(Z, colors='black', linewidths=0.8)
plt.clabel(cnt,[-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1], colors='black', fontsize=7)
plt.show()

# Ejercicio 12
import matplotlib.pyplot as plt
import numpy as np

n = 1024
X = np.random.normal(0, 1, n)
Y = np.random.normal(0, 1, n)

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.scatter(X, Y, c=np.arctan2(Y, X), cmap='rainbow', alpha=0.5, edgecolors='black', linewidths=0.5)
plt.xticks(())
plt.yticks(())
plt.show()

# Ejercicio 13

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mrk


class R2:
	def __init__(self, x=None, y=None):
		self.x = x
		self.y = y


class Pez:
	def __init__(self,x=None,y=None):
		self.pos = R2(x,y)
		self.vel = R2(x,y)
		self.maxVelPez = 15


class Cardumen:
	def __init__(self):
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
		# plt.show(block=False)  # Para hacer la simulación hay que descomentar estas 3 lineas
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
	# c.print() # descomentar esto para tener una simulación en movimiento
	if i == 0:  # comentar todos los if para la simulación
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

plt.show()  # comentar esto para la simulación

# Ejercicio 14    --> Se puede hacer más genérico, pero no me quedó tiempo para implementarlo mas lindo
import numpy as np
import matplotlib.pyplot as plt


class Persona:
	def __init__(self):
		self.mes = np.random.randint(1, 12)
		if self.mes in [1, 3, 5, 7, 8, 10, 12]:
			self.dia = np.random.randint(1, 31)
		elif self.mes in [4, 6, 9, 11]:
			self.dia = np.random.randint(1, 30)
		else:
			self.dia = np.random.randint(1, 28)


class Grupo:
	def __init__(self, tamanos_de_grupo):
		self.tam_grupo = np.random.choice(tamanos_de_grupo)
		self.grupo = []

		self.iniciar()
		self.cuenta = self.verificar()

	def iniciar(self):
		for i in range(self.tam_grupo):
			self.grupo.append(Persona())

	def verificar(self):
		for i in range(self.tam_grupo):
			for j in range(i+1, self.tam_grupo):
				if self.grupo[i].dia == self.grupo[j].dia and self.grupo[i].mes == self.grupo[j].mes:
					# self.cuenta += 1
					return 1
		return 0



# controlado por el usuario: Queda para más adelante intentar hacerlo más genérico de manera de poder cambiar los
# tamaños de grupo y que se acomode automáticamente
archivo = open('Estadistica_cumples.txt', 'w+')
tamanos_de_grupo = [10, 20, 30, 40, 50, 60]
niter = 1000
for n_experimientos in range(niter):
	mi_grupo = Grupo(tamanos_de_grupo)
	archivo.write(str(mi_grupo.tam_grupo)+' ' + str(mi_grupo.cuenta)+'\n')
archivo.close()

#  Armo la tabla:
tabla = open('Estadistica_cumples.txt', 'r')
tamano = []
repeticiones = []
for i in range(niter):
	lines = tabla.readline()
	# tupla.append(lines.split())
	tamano.append(int(lines.split()[0]))
	repeticiones.append(int(lines.split()[1]))

p10, p20, p30, p40, p50, p60 = 0,0,0,0,0,0
pp10, pp20, pp30, pp40, pp50, pp60 = 0,0,0,0,0,0
for i in range(niter):
	if repeticiones[i] == 1:
		if tamano[i] == 10:
			p10 += 1
		if tamano[i] == 20:
			p20 += 1
		if tamano[i] == 30:
			p30 += 1
		if tamano[i] == 40:
			p40 += 1
		if tamano[i] == 50:
			p50 += 1
		if tamano[i] == 60:
			p60 += 1
pp10 = p10/tamano.count(10)*100
pp20 = p20/tamano.count(20)*100
pp30 = p30/tamano.count(30)*100
pp40 = p40/tamano.count(40)*100
pp50 = p50/tamano.count(50)*100
pp60 = p60/tamano.count(60)*100
proba = [pp10, pp20, pp30, pp40, pp50, pp60]
tabla_resu = []
for i in range(2):
	tabla_resu.append(np.zeros(len(tamanos_de_grupo)))
tabla_resu[0] = tamanos_de_grupo
tabla_resu[1] = proba
print(tabla_resu)

# Hago un gráfico de barras:
label = [str('%.2f'%float(pp10)),str('%.2f'%float(pp20)),str('%.2f'%float(pp30)),str('%.2f'%float(pp40)),str('%.2f'%float(pp50)),str('%.2f'%float(pp60))]
plt.bar(tamanos_de_grupo, proba, width=3.5)
for i in range(len(tamanos_de_grupo)):
	plt.text(x=tamanos_de_grupo[i]-1.2, y=proba[i],s=label[i])
plt.xlabel("Tamaño de grupo")
plt.ylabel("Probabilidad de que dos personas cumplan años el mismo día")
plt.show()


# Ejercicio 15

import numpy as np


class Noiser:
	def __init__(self, minV, maxV):
		self._minV = minV
		self._maxV = maxV

	def __call__(self, x):
		argumento = x
		if type(argumento) is float:
			return self.sum_pseudo(x)
		else:
			return argumento

	def sum_pseudo(self, valor):
		pseudo = np.random.uniform(self._minV, self._maxV)
		return valor + pseudo


noiser = Noiser(-0.5, 0.5)

g = np.vectorize(noiser)
array = [1, 1.3, 3.4, 5]

print(g(array))


