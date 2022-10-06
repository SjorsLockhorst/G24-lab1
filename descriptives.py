import os

from extract import create_dialog_dataset

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = create_dialog_dataset()
    all_x = np.concatenate([x_train, x_test])
    all_y = np.concatenate([y_train, y_test])

    len_vect = np.vectorize(len)
    words = [sent.split() for sent in all_x]
    lengths = len_vect(np.array(words, dtype="object"))
    sorted_lens = np.sort(lengths)

    utt_fig, utt_ax = plt.subplots()
    utt_ax.hist(sorted_lens, bins=np.linspace(1, lengths.max()))
    utt_ax.set_title("Histrogram sentence utterence length.")
    utt_ax.set_xlabel("Utterance length in amount of words")
    utt_ax.set_ylabel("Frequency in the data")
    utt_ax.set_xticks(np.arange(1, lengths.max(), 1))

    # Create table and data to show label distribution
    training_tabel = PrettyTable(["Label", "N occurences"])
    values, n_occ = np.unique(all_y, return_counts=True)
    rows = np.column_stack((values, n_occ))
    training_tabel.add_rows(rows)

    vectorizer = CountVectorizer()
    vectorizer.fit(x_train)
    voc = vectorizer.vocabulary_

    test_vectorizer = CountVectorizer()
    test_vectorizer.fit(x_test)
    test_voc = test_vectorizer.vocabulary_

    n_out_of_voc = len(voc.keys() - test_voc.keys())
    print(f"Out of vocabulary words in test set: {n_out_of_voc}")

    print(f"Amount of unique labels: {len(values)}")
    print("Training data label distribution")
    print(training_tabel.get_string())
    PLOT_DIR = "plots"
    path = os.path.join(PLOT_DIR, "utterence_length_hist.png")
    utt_fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Average sentence length in words: M={lengths.mean()}, SD={lengths.std()}")
    print(f"Saved histogram of utterence lengths to {path}")
