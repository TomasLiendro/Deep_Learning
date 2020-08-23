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
