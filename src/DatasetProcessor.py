import re
import string
import sys

import pandas as pd
import numpy as np
import torchtext
from sklearn.datasets import load_files
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

import nltk

nltk.download('stopwords')
nltk.download('punkt')




class DatasetProcessor:
    def __init__(self):
        sys.stderr.write("Loading dataset from files ... ")
        self.data = self._get_labelled_dataframe()
        sys.stderr.write("Done!\nPreprocessing text ... ")
        self._preprocess_text()
        sys.stderr.write("Done!\n")
        self._vect_df = None
        self._pca_rep = None
        self._used_vect = ""

    def get_tf_idf_rep(self, max_f=10000):
        if self._used_vect != "tf-idf":
            self._pca_rep = None
            self._used_vect = "tf-idf"
            sys.stderr.write("Vectorizing text - TF-IDF ... ")
            tf_idf_vect = TfidfVectorizer(max_features=max_f)
            self._vect_df = tf_idf_vect.fit_transform(self.data["text"])
            self._vect_df = pd.DataFrame(self._vect_df.todense())
            sys.stderr.write("Done!\n")
        return self._vect_df

    def get_glove_rep(self, dimension=100):
        if self._used_vect != "glove":
            self._pca_rep = None
            self._used_vect = "glove"
            sys.stderr.write("Vectorizing text - GloVe ... ")
            glove = torchtext.vocab.GloVe(name="6B",  # trained on Wikipedia 2014 corpus of 6 billion words
                                          dim=dimension, cache="../data/glove/")

            self._vect_df = pd.DataFrame(columns=[ind for ind in range(dimension)])
            for i in range(self.data.shape[0]):
                doc_vect_rep = [0.0] * dimension
                for word in self.data.loc[i, "text"].split():
                    summing_array = np.array([doc_vect_rep, glove[word].tolist()])
                    doc_vect_rep = np.sum(summing_array, 0).tolist()
                self._vect_df.loc[i] = np.divide(doc_vect_rep, len(self.data.loc[i, "text"].split())).tolist()
            sys.stderr.write("Done!\n")
        return self._vect_df

    def get_pca_rep(self):
        if self._pca_rep is None:
            pca_algorithm = PCA(n_components=2, random_state=23)
            self._pca_rep = pca_algorithm.fit_transform(self._vect_df)
        return self._pca_rep

    def get_coordinates(self):
        return self._vect_df

    @staticmethod
    def _get_labelled_dataframe():
        labelled_dataset = load_files("../data/bbc-dataset/", encoding="utf-8")
        df = pd.DataFrame(list(zip(labelled_dataset["data"], labelled_dataset["target"])), columns=["text", "group_id"])
        group_names = labelled_dataset["target_names"]
        df["group_name"] = [group_names[group_id] for group_id in df["group_id"]]
        return df

    def _preprocess_text(self):
        # transform into lowercase
        self.data["text"] = self.data["text"].apply(lambda row: row.lower())
        # remove html tags
        self.data["text"] = self.data["text"].apply(lambda row: re.sub(r'<.*?>', '', row))
        # remove digits
        self.data["text"] = self.data["text"].apply(lambda row: re.sub(r'\d+', '', row))
        # remove punctuation
        trans_remove_punc = str.maketrans("", "", string.punctuation)
        self.data["text"] = self.data["text"].apply(lambda row: row.translate(trans_remove_punc))
        # remove diacritics
        self.data["text"] = self.data["text"].apply(lambda row: unidecode(row))
        # remove whitespaces
        self.data["text"] = self.data["text"].apply(lambda row: " ".join(row.split()))
        # remove stopwords (NLTK stopword list) while text stemming (using The Porter Stemming Algorithm)
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words("english"))
        self.data["text"] = self.data["text"].apply(lambda row: self._get_stems_without_stop_words(row, stop_words,
                                                                                                   stemmer))

    @staticmethod
    def _get_stems_without_stop_words(text: str, stop_words_set: set, stemmer):
        word_tokens = word_tokenize(text)
        without_stop_words = [stemmer.stem(word) for word in word_tokens if word not in stop_words_set]
        return " ".join(without_stop_words)
