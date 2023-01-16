import sklearn.cluster as clustering_algorithms


class KMeans:
    def __init__(self):
        self.clusters_id = None

    def run_kmeans(self, vectorized_data):
        kmeans = clustering_algorithms.KMeans(n_clusters=5, n_init=10, max_iter=500, random_state=0)
        kmeans = kmeans.fit(vectorized_data)
        self.clusters_id = kmeans.labels_
        return self.clusters_id


class OPTICS:
    def __init__(self):
        self.clusters_id = None

    def run_optics(self, vectorized_data):
        optics = clustering_algorithms.OPTICS(min_samples=8, algorithm="ball_tree", n_jobs=4)
        optics = optics.fit(vectorized_data)
        self.clusters_id = optics.labels_
        return self.clusters_id


class AgglomerativeClustering:
    def __init__(self):
        self.clusters_id = None

    def run_agglomerative_clustering(self, vectorized_data):
        ac = clustering_algorithms.AgglomerativeClustering(n_clusters=5, metric="euclidean", linkage="ward")
        ac = ac.fit(vectorized_data)
        self.clusters_id = ac.labels_
        return self.clusters_id
