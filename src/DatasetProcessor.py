import re
import string
import sys

import pandas as pd

try:
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.tokenize import word_tokenize
except:
    import nltk

    nltk.download('stopwords')
    nltk.download('punkt')

from sklearn.datasets import load_files
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode
import torch
import torchtext


class DatasetProcessor:
    def __init__(self):
        sys.stderr.write("Loading dataset from files ... ")
        self.data = self._get_labelled_dataframe()
        sys.stderr.write("Done!\nPreprocessing text ... ")
        self._preprocess_text()
        sys.stderr.write("Done!\n")
        self._vect_mtx = None

    def get_tf_idf_rep(self):
        sys.stderr.write("Vectorizing text ... ")
        tf_idf_vect = TfidfVectorizer(max_features=10000)
        self._vect_mtx = tf_idf_vect.fit_transform(self.data["text"])
        df = pd.DataFrame(self._vect_mtx.todense())
        sys.stderr.write("Done!\n")
        return df

    def get_glove_rep(self):
        sys.stderr.write("Vectorizing text ... ")
        glove = torchtext.vocab.GloVe(name="6B",  # trained on Wikipedia 2014 corpus of 6 billion words
                                      dim=50)
        print(glove["dog"])
        df = pd.DataFrame()
        sys.stderr.write("Done!\n")
        return df

    def get_pca_rep(self):
        pca_algorithm = PCA(n_components=2, random_state=23)
        return pca_algorithm.fit_transform(self._vect_mtx.toarray())

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
