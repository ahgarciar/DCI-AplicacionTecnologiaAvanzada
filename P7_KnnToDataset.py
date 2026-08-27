import pandas as pd
import numpy as np

df = pd.read_csv("iris/iris.csv")

columns = df.columns

entradas = df[columns[:-1]]
salidas = df[columns[-1]]

entradas = entradas.to_numpy()
salidas = salidas.to_numpy()

print("Entradas:")
print(entradas)
print("Salidas:")
print(salidas)

# KNN = K vecinos más cercanos
#scikit-learn
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5) # K = 5

neigh.fit(entradas, salidas)

#Obtiene la clase más probabable (moda) con base en la aplicación del knn
newCase = [[1.2, 10.4, 25.1, 1.4]]  ##<<<<-----
predicted_class = neigh.predict(newCase)
print(predicted_class)

#Calcula las probabilidades de que el individuo pertenezca a cada clase
print(neigh.predict_proba(newCase))


