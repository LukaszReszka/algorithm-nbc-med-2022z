import pandas as pd
import sklearn

from AlgorithmNBC import AlgorithmNBC
from DatasetProcessor import DatasetProcessor
from ResultsVisualiser import ResultsVisualiser

if __name__ == "__main__":
    dataset_processor = DatasetProcessor()
    nbc = AlgorithmNBC()
    results_visualiser = ResultsVisualiser(dataset_processor.data)

    # TF-IDF
    dataset_processor.get_tf_idf_rep()
    nbc.run(dataset_processor.get_tf_idf_rep(), 17)
    results_visualiser.visualise_results(dataset_processor.get_pca_rep(), nbc.clusters_id, "TF-IDF")

    # GloVe
    nbc.run(dataset_processor.get_glove_rep(), 14)
    results_visualiser.visualise_results(dataset_processor.get_pca_rep(), nbc.clusters_id, "GloVe")

    # test datasets
    coord, ids = sklearn.datasets.make_blobs(n_samples=2205, centers=5, random_state=0)
    nbc.run(pd.DataFrame(coord), 9)
    results_visualiser.visualise_test_dataset(coord, ids, nbc.clusters_id, 1)

    coord, ids = sklearn.datasets.make_blobs(n_samples=2205, centers=5, random_state=100)
    nbc.run(pd.DataFrame(coord), 20)
    results_visualiser.visualise_test_dataset(coord, ids, nbc.clusters_id, 2)

