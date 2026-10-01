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

def Kmeans_method(X, clusters):
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_  # Etiquetas de los clusters asignados
    centroids = kmeans.cluster_centers_  # Centros de los clusters

    print("Centroides finales:\n", centroids)
    print("\nAsignacion:\n", labels)

    # Graficar los puntos de datos con colores según sus clusters
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolors='k', alpha=0.7)

    # Graficar los centroides
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroides')

    plt.title("Clustering con K-Means")
    plt.xlabel("Característica 1")
    plt.ylabel("Característica 2")
    plt.legend()
    plt.show()

def Kmeans_method_3D(X, clusters):
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    print("Centroides finales:\n", centroids)
    print("\nAsignación de Clusters:\n", labels)

    # Gráfico 3D para visualizar los clusters
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', marker='o', alpha=0.7)
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='X', s=200, label='Centroides')

    ax.set_xlabel("Atributo 1")
    ax.set_ylabel("Atributo 2")
    ax.set_zlabel("Atributo 3")
    ax.set_title("Clustering con K-Means (3 Atributos)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X=loadData()
    #kmeans = Kmeans_method(X,3)
    kmeans = Kmeans_method_3D(X,clusters=3)
