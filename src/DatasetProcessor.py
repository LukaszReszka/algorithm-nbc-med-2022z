import string

import nltk
from sklearn.datasets import load_files
from unidecode import unidecode
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import re


class DatasetProcessor:
    def __init__(self):
        self.data = self._get_labelled_dataframe()
        self._preprocess_text()

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
        # remove stopwords (NLTK stopword list) while tokenizing text
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        stop_words = set(stopwords.words("english"))
        self.data["text"] = self.data["text"].apply(lambda row: self._remove_stop_words(row, stop_words))
        # conduct word stemming (using The Porter Stemming Algorithm)
        stemmer = PorterStemmer()
        self.data["text"] = self.data["text"].apply(lambda row: self._stem_word_tokens(row, stemmer))

    @staticmethod
    def _remove_stop_words(text: str, stop_words_set: set):
        word_tokens = word_tokenize(text)
        without_stop_words = [word for word in word_tokens if word not in stop_words_set]
        return without_stop_words

    @staticmethod
    def _stem_word_tokens(tokens: list[str], stemmer):
        return [stemmer.stem(word) for word in tokens]
