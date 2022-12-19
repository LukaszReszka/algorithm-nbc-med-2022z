from DatasetProcessor import DatasetProcessor
from NeighborsIndicator import NeighborsIndicator

if __name__ == "__main__":
    dataset_processor = DatasetProcessor()
    neighbors_ind = NeighborsIndicator(dataset_processor.get_vectorised_representation(), 5)
    print(neighbors_ind.get_knn(0))
    print(neighbors_ind.get_ndf(0))
