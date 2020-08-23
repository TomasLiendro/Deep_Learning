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
