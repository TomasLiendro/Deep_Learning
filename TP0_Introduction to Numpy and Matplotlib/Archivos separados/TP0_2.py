import numpy as np
import matplotlib.pyplot as plt
import scipy

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
# Para verificar los resultados anteriores uso la librería de Scipy
mean_teorico = k*th
std_teorico = np.sqrt(k*th**2)
print('Resultados teóricos')
print(mean_teorico, std_teorico)
plt.savefig('Ej2.pdf')
# plt.show()
