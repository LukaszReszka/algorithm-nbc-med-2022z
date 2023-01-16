import pandas as pd
import sklearn

from AlgorithmNBC import AlgorithmNBC
from ClusteringAlgorithms import OPTICS, AgglomerativeClustering, KMeans
from ClusteringEvaluator import ClusteringEvaluator
from DatasetProcessor import DatasetProcessor
from ResultsVisualiser import ResultsVisualiser

if __name__ == "__main__":
    dataset_processor = DatasetProcessor()
    nbc = AlgorithmNBC()
    agglomerative = AgglomerativeClustering()
    kmeans = KMeans()
    optics = OPTICS()
    results_visualiser = ResultsVisualiser(dataset_processor.data)
    clustering_eval = ClusteringEvaluator(dataset_processor.data)

    # TF-IDF
    # NBC #########################################
    nbc.run(dataset_processor.get_tf_idf_rep(), 17)
    results_visualiser.visualise_nbc_results(dataset_processor.get_pca_rep(), nbc.clusters_id, "TF-IDF")
    clustering_eval.evaluate_clustering(nbc.clusters_id, dataset_processor.get_coordinates(), "NBC (TF-IDF)")

    # KMeans ######################################
    results_visualiser.visualise_alg_results(dataset_processor.get_pca_rep(),
                                             kmeans.run_kmeans(dataset_processor.get_tf_idf_rep()), "K-Means", "TF-IDF")
    clustering_eval.evaluate_clustering(kmeans.clusters_id, dataset_processor.get_coordinates(), "KMeans (TF-IDF)")

    # OPTICS ######################################
    results_visualiser.visualise_alg_results(dataset_processor.get_pca_rep(),
                                             optics.run_optics(dataset_processor.get_tf_idf_rep()), "OPTICS", "TF-IDF")
    clustering_eval.evaluate_clustering(optics.clusters_id, dataset_processor.get_coordinates(), "OPTICS (TF-IDF)")

    # Agglomerative Clustering ####################
    results_visualiser.visualise_alg_results(dataset_processor.get_pca_rep(),
                                             agglomerative.run_agglomerative_clustering(
                                                 dataset_processor.get_tf_idf_rep()), "Agglomerative Clustering",
                                             "TF-IDF")
    clustering_eval.evaluate_clustering(agglomerative.clusters_id, dataset_processor.get_coordinates(),
                                        "Agglomerative (TF-IDF)")

    # GloVe
    # NBC #########################################
    nbc.run(dataset_processor.get_glove_rep(), 14)
    results_visualiser.visualise_nbc_results(dataset_processor.get_pca_rep(), nbc.clusters_id, "GloVe")
    clustering_eval.evaluate_clustering(nbc.clusters_id, dataset_processor.get_coordinates(), "NBC (GloVe)")

    # KMeans ######################################
    results_visualiser.visualise_alg_results(dataset_processor.get_pca_rep(),
                                             kmeans.run_kmeans(dataset_processor.get_glove_rep()), "K-Means", "GloVe")
    clustering_eval.evaluate_clustering(kmeans.clusters_id, dataset_processor.get_coordinates(), "KMeans (GloVe)")

    # OPTICS ######################################
    results_visualiser.visualise_alg_results(dataset_processor.get_pca_rep(),
                                             optics.run_optics(dataset_processor.get_glove_rep()), "OPTICS", "GloVe")
    clustering_eval.evaluate_clustering(optics.clusters_id, dataset_processor.get_coordinates(), "OPTICS (GloVe)")

    # Agglomerative Clustering ####################
    results_visualiser.visualise_alg_results(dataset_processor.get_pca_rep(),
                                             agglomerative.run_agglomerative_clustering(
                                                 dataset_processor.get_glove_rep()), "Agglomerative Clustering",
                                             "GloVe")
    clustering_eval.evaluate_clustering(agglomerative.clusters_id, dataset_processor.get_coordinates(),
                                        "Agglomerative (GloVe)")

    # test datasets
    # TEST 1
    coord, ids = sklearn.datasets.make_blobs(n_samples=2205, centers=5, random_state=0)
    nbc.run(pd.DataFrame(coord), 9)
    results_visualiser.visualise_test_dataset(coord, ids, nbc.clusters_id, 1)
    clustering_eval.evaluate_clustering(nbc.clusters_id, coord, "NBC (TEST1)", ids)

    # TEST 2
    coord, ids = sklearn.datasets.make_blobs(n_samples=2205, centers=5, random_state=100)
    nbc.run(pd.DataFrame(coord), 20)
    results_visualiser.visualise_test_dataset(coord, ids, nbc.clusters_id, 2)
    clustering_eval.evaluate_clustering(nbc.clusters_id, coord, "NBC (TEST2)", ids)
