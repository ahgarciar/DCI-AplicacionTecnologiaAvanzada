import matplotlib.pyplot as plt
from datasets import load_dataset
from fontTools.misc.cython import returns
from scipy.cluster.vq import kmeans
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score


def loadData():
    iris = load_iris()
    X = iris.data[:, :4]
    return X

def Import_CSV(csv_file):
    import pandas as pd
    df = pd.read_csv(csv_file)
    X = df.values
    return X

def Kmeans_PCA(X, clusters):
    from sklearn.decomposition import PCA
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    print("Centroides finales:\n", centroids)
    print("\nAsignación de Clusters:\n", labels)

    # Opcional: Reducir a 2D con PCA si hay más de 2 atributos
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        # Mostrar los componentes principales
        print("Componentes principales:\n", pca.components_)
        #print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)
        #print("Varianza total explicada:", sum(pca.explained_variance_ratio_))

        centroids_pca = pca.transform(centroids)

        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Centroides')
        plt.xlabel("Componente Principal 1")
        plt.ylabel("Componente Principal 2")
        plt.title("Clusters en 2D usando PCA")
        plt.legend()
        plt.show()

def Kmeans_PCA3D(X, clusters):
    from sklearn.decomposition import PCA
    # Aplicar K-Means
    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    print("Centroides finales:\n", centroids)
    print("\nAsignación de Clusters:\n", labels)

    # Opcional: Reducir a 3D con PCA si hay más de 3 atributos
    if X.shape[1] > 3:
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)

        # Mostrar los componentes principales
        print("Componentes principales:\n", pca.components_)
        #print("Varianza explicada por cada componente:", pca.explained_variance_ratio_)
        #print("Varianza total explicada:", sum(pca.explained_variance_ratio_))

        centroids_pca = pca.transform(centroids)

        # Gráfico 3D para visualizar los componentes
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, cmap='viridis',marker='o', alpha=0.7)
        ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], centroids_pca[:, 2], c='red', marker='X', s=200, label='Centroides')
        ax.set_xlabel("CP 1")
        ax.set_ylabel("CP 2")
        ax.set_zlabel("CP 3")
        ax.set_title("Clustering con K-Means PCA 3D")
        plt.legend()
        plt.show()

def pca():
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    X = iris.data[:, :4]

    # Normalizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aplicar PCA para reducir a 2 dimensiones
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Ver los nuevos datos transformados
    print(X_pca[:5])  # Primeras 5 filas después del PCA


if __name__ == '__main__':
    X=loadData()
    kmeans = Kmeans_PCA(X, 3)
    kmeans = Kmeans_PCA3D(X, 3)

