import numpy as np
import matplotlib.pyplot as plt

def f(x,y):
	return (1 - x / 2 + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)


n = 256
x = np.linspace(-3, 3, 4 * n)
y = np.linspace(-3, 3, 3 * n)
X,Y = np.meshgrid(x,y)

plt.contourf(f(X, Y),alpha=.7, cmap='hot')
C = plt.contour(f(X, Y), colors='black')
plt.clabel(C, [-0.5,-0.25,0,0.25,0.5,0.75,1])

plt.xticks(())
plt.yticks(())
plt.show()