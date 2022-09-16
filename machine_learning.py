import os
import pickle

import numpy as np

from baseline import evaluate
from extract import create_dataset
from vectorize import create_bag_of_words, vectorize_all


def load_model():
    DIR = "models/"
    LOG_REG_PATH = os.path.join(DIR, "log_reg.pickle")
    VOC_PATH = os.path.join(DIR, "vocabulary.pickle")
    with open(LOG_REG_PATH, "rb") as file:
        model = pickle.load(file)
    with open(VOC_PATH, "rb") as file:
        vocabulary = pickle.load(file)
    return model, vocabulary


def evaluate_model():
    _, _, x_test, y_test = create_dataset()
    log_reg, vocabulary = load_model()
    x_vectorized = vectorize_all(x_test, vocabulary)
    res = log_reg.predict(x_vectorized)
    correct = evaluate(y_test, res)
    print(correct)


if __name__ == "__main__":
    evaluate_model()
