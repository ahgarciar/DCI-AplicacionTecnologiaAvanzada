import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

def loadData():
    iris = load_iris()
    X = iris.data[:, :4] #excluye la clase
    return X

def Import_CSV(csv_file):
    import pandas as pd
    df = pd.read_csv(csv_file)
    X = df.values #tendría que no tener la clase el archivo que se lee , o quitarla despues de leer el archivo
    return X

def ElbowMethod(X):
    # probamos diferentes valores de k para visualizar el metodo del codo
    wcss = []
    k_values = range(1, 11) #valores posibles de k  ... 11 se puede cambiar por la raiz cuadrada del total de elementos de X

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        #print(kmeans.labels_)
        wcss.append(kmeans.inertia_)  # WCSS - valor dispersion
        print("K= ", k)
        print("WCSS:", kmeans.inertia_)

    # Graficar el metodo del codo (Elbow Method)
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, wcss, marker='o', linestyle='-')
    plt.xlabel('Número de Clústeres (k)')
    plt.ylabel('WCSS')
    plt.title('Método Elbow para encontrar k óptimo')
    plt.show()

def Metrics(X, clusters):
    # WCSS - Inercia
    # Valor bajo - Mejor agrupacion
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    print("WCSS:", kmeans.inertia_)

    # Silhouette
    # Cercano a uno buena separacion
    # Cercano a cero mala separacion
    # Negativo MAL
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    print("Coeficiente de Silhouette:", silhouette_avg)

    # Davies-Bouldin
    # Bajo - Mejor separacion
    #Alto - Mala separacion
    from sklearn.metrics import davies_bouldin_score
    db_index = davies_bouldin_score(X, kmeans.labels_)
    print("Índice de Davies-Bouldin:", db_index)



if __name__ == '__main__':
    X = loadData()
    ElbowMethod(X)
    print()
    print("Con 2 Clusters:")
    Metrics(X, 2)
    print("Con 3 Clusters:")
    Metrics(X, 3)
