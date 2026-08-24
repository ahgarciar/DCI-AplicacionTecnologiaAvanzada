
import pandas as pandita
import numpy as np
import matplotlib.pyplot as plt

df = pandita.read_csv("iris/iris.csv")
          #,header=None)

print(df)

#pendiente...
#df = df["class"]

columns = df.columns
print(columns)

##quitar las columnas que no son númericas..!

for i in range(columns.size-1):
    print(columns[i])
    #obtener la informacion de la columna sepal length
    sepal_length = df[columns[i]]

    valor_min = min(sepal_length)
    valor_max = max(sepal_length)
    valor_promedio = np.mean(sepal_length)
    valor_std = np.std(sepal_length)

    print("Valor menor: " + str(valor_min))
    print("Valor mayor: " + str(valor_max))
    print("Valor promedio: " + str(valor_promedio))
    print("Valor std: " + str(valor_std))

    print("##################################")

    df.boxplot(column=columns[i])
    plt.show()

