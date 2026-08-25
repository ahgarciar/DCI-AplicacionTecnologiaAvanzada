
#Valores de entrada
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]

print("X:")
print(X)
print("y:")
print(y)

# KNN = K vecinos más cercanos
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3) # K = 3

neigh.fit(X, y)

#Obtiene la clase más probabable (moda) con base en la aplicación del knn
newCase = [[1.1]]
predicted_class = neigh.predict(newCase)
print(predicted_class)

#Calcula las probabilidades de que el individuo pertenezca a cada clase
print(neigh.predict_proba([[0.9]]))


