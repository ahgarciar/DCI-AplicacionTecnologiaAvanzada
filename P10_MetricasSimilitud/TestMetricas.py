from P10_MetricasSimilitud import Metricas as m

#sepal length	sepal width	petal length	petal width	class
vectorA = [5.1,	3.5,	1.4,	0.2]	# Iris-setosa
vectorB = [4.9,	  3,	1.4,	0.2]	# Iris-setosa
vectorC = [5.4,	3.4,	1.5,	0.4]	#Iris-setosa

vectorD = [5.7,	2.5,	5,	2]	# Iris-virginica

vectorE = [7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4,"5"]
vectorF = [7.8,0.88,0,2.6,0.098,25,67,0.9968,3.2,0.68,9.8,"5"]

tipo_metrica = 11

dist = m.getDistancia(vectorA, vectorB, tipo= tipo_metrica)
print(dist)
dist = m.getDistancia(vectorA, vectorC, tipo= tipo_metrica)
print(dist)
dist = m.getDistancia(vectorB, vectorC, tipo= tipo_metrica)
print(dist)

print("\nComparacion 2: ")
dist = m.getDistancia(vectorD, vectorA, tipo= tipo_metrica)
print(dist)
dist = m.getDistancia(vectorD, vectorB, tipo= tipo_metrica)
print(dist)
dist = m.getDistancia(vectorD, vectorC, tipo= tipo_metrica)
print(dist)


