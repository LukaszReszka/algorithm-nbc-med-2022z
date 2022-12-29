from tqdm import trange

from NeighborsIndicator import NeighborsIndicator


class AlgorithmNBC:
    def __init__(self):
        self.clusters_id = None

    def run(self, vect_rep_df, k: int):
        neighborhood = NeighborsIndicator(vect_rep_df, k)
        self.clusters_id = [-1] * neighborhood.tf_idf_df.shape[0]
        current_gr_id = 0

        for p_id in trange(neighborhood.tf_idf_df.shape[0], desc="Clustering points"):
            if self.clusters_id[p_id] == -1 and neighborhood.get_ndf(p_id) >= 1:
                self.clusters_id[p_id] = current_gr_id
                seed = []

                for neighbor_id in neighborhood.get_knn(p_id):
                    self.clusters_id[neighbor_id] = current_gr_id
                    if neighborhood.get_ndf(neighbor_id) >= 1:
                        seed.append(neighbor_id)

                while seed:
                    for neighbor_id in neighborhood.get_knn(seed.pop()):
                        if self.clusters_id[neighbor_id] == -1:
                            self.clusters_id[neighbor_id] = current_gr_id
                            if neighborhood.get_ndf(neighbor_id) >= 1:
                                seed.append(neighbor_id)

                current_gr_id += 1
