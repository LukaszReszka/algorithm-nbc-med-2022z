from DatasetProcessor import DatasetProcessor
from AlgorithmNBC import AlgorithmNBC

if __name__ == "__main__":
    dataset_processor = DatasetProcessor()
    nbc = AlgorithmNBC()
    nbc.run(dataset_processor.get_tf_idf_rep(), 20)
    print(set(nbc.clusters_id))
