import sys

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


class ResultsVisualiser:
    def __init__(self, data):
        self.data = data

    def visualise_results(self, pca_rep, clusters_ids, vect_type: str):
        sys.stderr.write("Visualising results ... ")
        self._visualise_clusters("Klastry referencyjne (" + vect_type + ")", self.data["group_name"],
                                 pca_rep)
        self._visualise_clusters("Klastry wyznaczone przez NBC (" + vect_type + ")", clusters_ids,
                                 pca_rep)

        sys.stderr.write("Done - saved to 'visuals' folder!")

    @staticmethod
    def _visualise_clusters(plot_title: str, group_names: list, pca_rep):
        df_to_vis = pd.DataFrame({"X1": pca_rep[:, 0], "X2": pca_rep[:, 1], "Grupa": group_names})

        plt.switch_backend("Agg")
        fig = plt.figure(figsize=(12, 8), facecolor="w")
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("X1", fontdict={"weight": "bold", "size": 11})
        ax.set_ylabel("X2", fontdict={"weight": "bold", "size": 11})
        fig.suptitle(plot_title, fontweight="bold")
        sns.scatterplot(data=df_to_vis, x="X1", y="X2", hue="Grupa", palette="tab10")
        plt.savefig("../visuals/" + plot_title + ".png", bbox_inches="tight")
        plt.close(fig)
