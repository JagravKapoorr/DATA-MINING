from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Function to perform K-means clustering and compute MSE after each iteration
def kmeans_clustering(X, n_clusters, max_iter):
    kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=0)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    mse = []
    for i in range(1, max_iter + 1):
        kmeans = KMeans(n_clusters=n_clusters, max_iter=i, random_state=0)
        kmeans.fit(X)
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_
        mse.append(mean_squared_error(X, centroids[labels]))
    return mse

# Load the Iris dataset
iris = load_iris()
X = iris.data

# Perform K-means clustering with different parameters
n_clusters_list = [2, 3, 4, 5]  # Number of clusters
max_iter_list = [5, 10, 15, 20]  # Maximum number of iterations

plt.figure(figsize=(12, 8))
for n_clusters in n_clusters_list:
    for max_iter in max_iter_list:
        mse = kmeans_clustering(X, n_clusters, max_iter)
        plt.plot(range(1, max_iter + 1), mse, label=f'n_clusters={n_clusters}, max_iter={max_iter}')

plt.title('Mean Squared Error (MSE) after each iteration for Iris Dataset')
plt.xlabel('Number of iterations')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.show()
