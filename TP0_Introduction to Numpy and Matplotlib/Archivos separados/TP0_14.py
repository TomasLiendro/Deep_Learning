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
