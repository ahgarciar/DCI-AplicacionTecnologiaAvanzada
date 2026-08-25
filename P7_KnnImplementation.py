
#Valores de entrada
X = [
[24,	20],
[39,	7],
[48,	35],
[1, 16],
[22,	40],
[34,	1],
[50,	23],
[11,	5],
[39,	26],
[18,	44],
[-44,	22],
[-36,	21],
[-31,	41],
[-38,	49],
[-25,	49],
[-10,	49],
[-15,	27],
[-42,	45],
[-37,	25],
[-13,	29],
[-9,	-26],
[-38,	-9],
[-43,	-37],
[-46,	-47],
[-36,	-41],
[-29,	-29],
[-42,	-31],
[-16,	-20],
[-4,	-6],
[-44,	-18],
]
y = [
"A",
"A",
"A",
"A",
"A",
"A",
"A",
"A",
"A",
"A",
"B",
"B",
"B",
"B",
"B",
"B",
"B",
"B",
"B",
"B",
"C",
"C",
"C",
"C",
"C",
"C",
"C",
"C",
"C",
"C"
]

print("X:")
print(X)
print("y:")
print(y)

# KNN = K vecinos más cercanos
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5) # K = 5

neigh.fit(X, y)

#Obtiene la clase más probabable (moda) con base en la aplicación del knn
newCase = [[-3, -1]]
predicted_class = neigh.predict(newCase)
print(predicted_class)

#Calcula las probabilidades de que el individuo pertenezca a cada clase
print(neigh.predict_proba(newCase))


