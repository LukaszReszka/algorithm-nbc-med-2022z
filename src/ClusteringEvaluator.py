from sklearn.metrics import rand_score, silhouette_score


class ClusteringEvaluator:
    def __init__(self, data):
        self.true_labels = data["group_name"]
        with open("../eval_metrics.txt", "w", encoding="utf-8"):
            pass

    def evaluate_clustering(self, clusters_id, coordinates, name: str, ids=None):
        labels_true = self.true_labels if ids is None else ids
        with open("../eval_metrics.txt", "a", encoding="utf-8") as f:
            f.write(name + " - Rand: " + str(rand_score(labels_true, clusters_id)) + "\n")
            f.write(name + " - Silhouette: " + str(
                silhouette_score(coordinates, labels=clusters_id, metric="euclidean")) + "\n")
