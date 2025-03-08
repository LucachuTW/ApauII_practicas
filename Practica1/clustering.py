import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

def kmeans(X, n_clusters, max_iter=100, init_method='random_points'):
    """Implementación do algoritmo K-Means.

    Args:
        X (numpy.ndarray): Datos de entrada (n_samples, n_features).
        n_clusters (int): Número de clusters a formar.
        max_iter (int, optional): Número máximo de iteracións. Defaults to 100.
        init_method (str, optional): Método de inicialización dos centroides ('random_points' ou 'random_data'). Defaults to 'random_points'.

    Returns:
        tuple: (centroides, etiquetas)
    """

    n_samples, n_features = X.shape

    # Inicialización dos centroides
    if init_method == 'random_points':
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        centroids = np.random.uniform(min_vals, max_vals, (n_clusters, n_features))
    elif init_method == 'random_data':
        indices = np.random.choice(n_samples, n_clusters, replace=False)
        centroids = X[indices]
    else:
        raise ValueError("init_method debe ser 'random_points' ou 'random_data'")

    # Iteracións
    for _ in range(max_iter):
        # Asignación de puntos a clusters
        distances = pairwise_distances(X, centroids)
        labels = np.argmin(distances, axis=1)

        # Equilibrio de tamaño
        for i in range(n_samples):
            equal_dist_centroids = np.where(distances[i] == distances[i][labels[i]])[0]
            if len(equal_dist_centroids) > 1:
                cluster_sizes = np.bincount(labels)
                labels[i] = equal_dist_centroids[np.argmin(cluster_sizes[equal_dist_centroids])]

        # Actualización de centroides
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(n_clusters)])

        # Converxencia
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, labels

# Xeración de datos de exemplo
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Aplicación de K-Means
n_clusters = 4
centroids, labels = kmeans(X, n_clusters, init_method='random_points')

# Visualización de resultados
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='x')
plt.title('K-Means Clustering')
plt.show()

# Probamos con otro numero de clusters
n_clusters = 3
centroids, labels = kmeans(X, n_clusters, init_method='random_data')

# Visualización de resultados
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='x')
plt.title('K-Means Clustering')
plt.show()

# Probamos con inicializacion aleatoria de puntos en vez de puntos del dataset
n_clusters = 4
centroids, labels = kmeans(X, n_clusters, init_method='random_points')

# Visualización de resultados
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='x')
plt.title('K-Means Clustering')
plt.show()