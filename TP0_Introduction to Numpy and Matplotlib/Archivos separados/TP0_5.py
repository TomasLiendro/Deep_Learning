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
