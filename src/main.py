from AlgorithmNBC import AlgorithmNBC
from DatasetProcessor import DatasetProcessor
from ResultsVisualiser import ResultsVisualiser

if __name__ == "__main__":
    dataset_processor = DatasetProcessor()

    nbc = AlgorithmNBC()
    results_visualiser = ResultsVisualiser(dataset_processor.data)

    nbc.run(dataset_processor.get_tf_idf_rep(), 17)
    results_visualiser.visualise_results(dataset_processor.get_pca_rep(), nbc.clusters_id, "TF-IDF")

    nbc.run(dataset_processor.get_glove_rep(), 14)
    results_visualiser.visualise_results(dataset_processor.get_pca_rep(), nbc.clusters_id, "GloVe")
