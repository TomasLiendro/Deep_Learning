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


plot_parabola(1, 1, -2)  # llamado a la funcion de plot


