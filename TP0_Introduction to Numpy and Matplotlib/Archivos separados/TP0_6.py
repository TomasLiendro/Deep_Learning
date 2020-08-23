from TP0_5 import Lineal


class Exponential(Lineal):
	def __call__(self, x=0):
		return self.a * x ** self.b

# '''
# Uncomment this:
equation = Exponential(1, 2)
solution = equation(3)
print(solution)
# '''