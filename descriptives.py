import os

from extract import create_dialog_dataset

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = create_dialog_dataset()

    len_vect = np.vectorize(len)
    words = [sent.split() for sent in x_train]
    lengths = len_vect(np.array(words, dtype="object"))
    sorted_lens = np.sort(lengths)

    plt.title("Histrogram training data utterence length.")
    plt.hist(sorted_lens, bins=np.linspace(1, lengths.max()))
    plt.xlabel("Utterance length in amount of words")
    plt.ylabel("Frequency in the data")

    # Create table and data to show label distribution
    training_tabel = PrettyTable(["Label", "N occurences"])
    values, unique = np.unique(y_train, return_counts=True)
    rows = np.column_stack((values, unique))
    training_tabel.add_rows(rows)

    vectorizer = CountVectorizer()
    vectorizer.fit(x_train)
    voc = vectorizer.vocabulary_

    test_vectorizer = CountVectorizer()
    test_vectorizer.fit(x_test)
    test_voc = test_vectorizer.vocabulary_

    n_out_of_voc = len(voc.keys() - test_voc.keys())
    print(f"Out of vocabulary words in test set: {n_out_of_voc}")

    print("Training data label distribution")
    print(training_tabel.get_string())
    PLOT_DIR = "plots"
    path = os.path.join(PLOT_DIR, "utterence_length_training_data.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved histogram of utterence lengths to {path}")
