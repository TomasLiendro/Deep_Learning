import p0_lib
from p0_lib import rectangulo
from p0_lib.circunferencia import PI, area
from p0_lib.elipse import area
from p0_lib.rectangulo import area as area_rect

print(p0_lib.rectangulo.area(1, 2))
print(rectangulo.area(1, 2))
print(area(1,2))  # --> Es el área de la elipse. El area de circ. queda pisada
print(area_rect(1, 2))
