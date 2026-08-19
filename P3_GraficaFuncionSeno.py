import math

from matplotlib import pyplot as plt

# y = mx + b

x = [i for i in range(-10, 11, 1)]
y = []

import math as m

for valor in x:
    res = m.sin(valor)
    y.append(res)

print(x)
print(y)

plt.plot(x, y)
plt.show()
