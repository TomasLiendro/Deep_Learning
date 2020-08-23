import circunferencia as circle
from circunferencia import PI, area
from numpy import pi

if circle.PI is pi:
	solution1 = circle.area(10)
	print('%3.f' % solution1)
else:
	Exception

if PI is pi:
	solution2 = area(10)
	print('%3.f' % solution2)
else:
	Exception

print('They are the same object!' if solution1 is solution2 else 'They are different objects!')
print(id(solution1), id(solution2))
# No son el mismo objeto: Falta buscar la definición