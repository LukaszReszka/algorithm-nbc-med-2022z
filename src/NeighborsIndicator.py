from sklearn.neighbors import NearestNeighbors
from tqdm import trange


class NeighborsIndicator:
    def __init__(self, vectorised_sparse_matrix, k: int):
        self.tf_idf_df = vectorised_sparse_matrix
        self._knn, self._rknn_numb = dict(), dict()
        self._get_nearest_neighbors(k)
        self._ndf = self._calculate_ndf()

    def get_knn(self, point_id: int) -> list[int]:
        return self._knn[point_id]

    def get_ndf(self, point_id: int) -> int:
        return self._ndf[point_id]

    def _calculate_ndf(self):
        points_ndf = dict()
        for i in trange(self.tf_idf_df.shape[0], desc="Calculating NDF"):
            if i in self._rknn_numb:
                points_ndf[i] = self._rknn_numb[i] / len(self._knn[i])
            else:
                points_ndf[i] = 0.0

        return points_ndf

    def _get_nearest_neighbors(self, k: int):
        nb = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(self.tf_idf_df)

        for i in trange(self.tf_idf_df.shape[0], desc="Calculating (R)kNN"):
            point_coordinates = self.tf_idf_df.loc[i].to_numpy().reshape(1, -1)
            distances, _ = nb.kneighbors(point_coordinates)  # not k+NN, so radius_neighbors needed
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
