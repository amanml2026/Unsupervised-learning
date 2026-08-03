from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from kmeans import KMeans_Clustering

X,Y = make_blobs(
    n_samples=500,      # Total number of samples
    n_features=2,       # Two features
    centers=5,          # Number of clusters
    cluster_std=1.0,    # Standard deviation of each cluster
    random_state=42
)

# plt.scatter(X[:,0],X[:,1])
# plt.show()

km = KMeans_Clustering(n_clusters=5,max_iters=600)
y_pred = km.fit_predict(X)

plt.scatter(X[:,0],X[:,1],c=y_pred)
plt.show()