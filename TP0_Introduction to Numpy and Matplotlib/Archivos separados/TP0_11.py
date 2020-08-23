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