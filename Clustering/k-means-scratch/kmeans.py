import random
import numpy as np

class KMeans_Clustering:
    def __init__(self,max_iters=100,n_clusters=2):
        self.max_iters = max_iters
        self.n_clusters = n_clusters
        self.centroids = None

    def fit_predict(self,X):
        random_idx = random.sample(range(X.shape[0]),self.n_clusters)
        self.centroids = X[random_idx]

        for _ in range(self.max_iters):
            # assign clusters
            cluster_group = self._assign_clusters(X)
            # updating the centroids
            old_centroids = self.centroids.copy()
            new_centroids = self._new_centroids(X,cluster_group)
            # finish
            if (old_centroids == new_centroids).all():
                break
            self.centroids = new_centroids

        return cluster_group

    def _assign_clusters(self,X):
        distances = []
        cluster_group = []
        for row in X:
            for ctr in self.centroids:
                distances.append(np.linalg.norm(row-ctr))
            min_distance = min(distances)
            idx = distances.index(min_distance)
            cluster_group.append(idx)
            distances.clear()

        return np.array(cluster_group)

    def _new_centroids(self,X,cluster_group):
        cluster_labels = np.unique(cluster_group)
        new_centroids = []

        for label in cluster_labels:
            new_centroids.append(X[cluster_group == label].mean(axis=0))

        return np.array(new_centroids)
