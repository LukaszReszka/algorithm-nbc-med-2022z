import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class ResultsVisualiser:
    def __init__(self, data, pca_rep, clusters_id):
        self.data = data
        self.pca_rep = pca_rep
        self.clusters_id = clusters_id
        self._visualise_results()

    def _visualise_results(self):
        sys.stderr.write("Visualising results ... ")
        self._visualise_clusters(self.data["group_name"], "Klastry referencyjne")
        self._visualise_clusters(self.clusters_id, "Klastry wyznaczone przez NBC")
        sys.stderr.write("Done - saved to 'visuals' folder!")

    def _visualise_clusters(self, group_names: list, title: str):
        df_to_vis = pd.DataFrame({"X1": self.pca_rep[:, 0], "X2": self.pca_rep[:, 1], "Grupa": group_names})

        plt.switch_backend("Agg")
        fig = plt.figure(figsize=(12, 8), facecolor="w")
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("X1", fontdict={"weight": "bold", "size": 11})
        ax.set_ylabel("X2", fontdict={"weight": "bold", "size": 11})
        fig.suptitle(title, fontweight="bold")
        sns.scatterplot(data=df_to_vis, x="X1", y="X2", hue="Grupa", palette="tab10")
        plt.savefig("../visuals/" + title + ".png", bbox_inches="tight")
        plt.close(fig)
