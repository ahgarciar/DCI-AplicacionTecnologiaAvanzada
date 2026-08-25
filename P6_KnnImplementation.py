
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

newCase = [[1.1]]
predicted_class = neigh.predict(newCase)
print(predicted_class)

print(neigh.predict_proba([[0.9]]))

