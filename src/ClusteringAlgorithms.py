import sklearn.cluster as clustering_algorithms


def run_kmeans(vectorized_data):
    kmeans = clustering_algorithms.KMeans(n_clusters=5, n_init=10, max_iter=500, random_state=0)
    kmeans = kmeans.fit(vectorized_data)
    return kmeans.labels_


def run_optics(vectorized_data):
    optics = clustering_algorithms.OPTICS(min_samples=8, algorithm="ball_tree", n_jobs=4)
    optics = optics.fit(vectorized_data)
    return optics.labels_


def run_agglomerative_clustering(vectorized_data):
    ac = clustering_algorithms.AgglomerativeClustering(n_clusters=5, metric="euclidean", linkage="ward")
    ac = ac.fit(vectorized_data)
    return ac.labels_
