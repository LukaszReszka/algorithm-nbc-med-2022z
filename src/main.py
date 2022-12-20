from AlgorithmNBC import AlgorithmNBC
from DatasetProcessor import DatasetProcessor
from ResultsVisualiser import ResultsVisualiser

if __name__ == "__main__":
    dataset_processor = DatasetProcessor()

    nbc = AlgorithmNBC()
    nbc.run(dataset_processor.get_tf_idf_rep(), 17)

    results_visualiser = ResultsVisualiser(dataset_processor.data, dataset_processor.get_pca_rep(), nbc.clusters_id)
