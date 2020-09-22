import numpy as np
import TP6.create_struct as create_netowrk
from itertools import product

dim = 10
x_train = np.array([i for i in product([-1, 1], repeat=dim)])
y_train = np.where(np.prod(x_train, axis=1) == 1, 1, -1)[:, np.newaxis]

create_netowrk.create_struct(x_train, y_train)
# print(x_train, y_train)