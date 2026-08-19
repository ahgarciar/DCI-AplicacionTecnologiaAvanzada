
from matplotlib import pyplot as plt

# y = mx + b

x = [i for i in range(-10, 11, 1)]
y = []

for valor in x:
    res = 5 * valor + 2
    y.append(res)

print(x)
print(y)

plt.plot(x, y)
plt.show()
