import sklearn.cluster as clustering_algorithms


def run_kmeans(vectorized_data):
    kmeans = clustering_algorithms.KMeans(n_clusters=5, n_init=10, max_iter=500, random_state=0)
    kmeans = kmeans.fit(vectorized_data)
    return kmeans.labels_
