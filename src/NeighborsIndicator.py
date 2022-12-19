from sklearn.neighbors import NearestNeighbors


class NeighborsIndicator:
    def __init__(self, vectorised_sparse_matrix, k: int):
        self.tf_idf_df = vectorised_sparse_matrix
        self._knn, self._rknn_numb = dict(), dict()
        self._get_nearest_neighbors(k)
        self._ndf = dict()

    def _get_nearest_neighbors(self, k: int):
        nb = NearestNeighbors(n_neighbors=k + 1, algorithm="kd_tree").fit(self.tf_idf_df)

        # self.tf_idf_df.shape[0]
        for i in range(1):
            point_coordinates = self.tf_idf_df.loc[i].to_numpy().reshape(1, -1)
            distances, _ = nb.kneighbors(point_coordinates)  # not k+NN, so radius_neighbors needed

            if len(distances[0]) > 1:
                eps = max(distances[0]) + 0.00000001
                point_neighbors = nb.radius_neighbors(point_coordinates, radius=eps, return_distance=False)
                point_neighbors = (point_neighbors[0].tolist())
                point_neighbors.remove(i)
                self._knn[i] = point_neighbors
                for p in point_neighbors:
                    if p in self._rknn_numb:
                        self._rknn_numb[p] += 1
                    else:
                        self._rknn_numb[p] = 1
