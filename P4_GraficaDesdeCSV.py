
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("BookFuncionSeno.csv", encoding="utf-8")


#print(df)
#print(df.head())
#print(df.head(1))

print(df.columns)

x = df["X"]
print(x)

tabla = df.to_numpy()

x= tabla[:,0]
y = tabla[:,1]

plt.plot(x, y)
plt.show()

print()