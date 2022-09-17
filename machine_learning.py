import os
import pickle

from baseline import evaluate
from extract import create_dataset
from vectorize import vectorize_all


def load_model():
    """Load model and matching vocabulary."""
    DIR = "models/"
    LOG_REG_PATH = os.path.join(DIR, "log_reg.pickle")
    VOC_PATH = os.path.join(DIR, "vocabulary.pickle")

    with open(LOG_REG_PATH, "rb") as file:
        model = pickle.load(file)
    with open(VOC_PATH, "rb") as file:
        vocabulary = pickle.load(file)
    return model, vocabulary


def evaluate_log_reg():
    """Evaluate model based on test set"""
    _, x_test, _, y_test = create_dataset()
    log_reg, vocabulary = load_model()
    x_vectorized = vectorize_all(x_test, vocabulary)
    res = log_reg.predict(x_vectorized)
    correct = evaluate(y_test, res)
    print(f"Correct: {correct:.2f}%")


if __name__ == "__main__":
    print("Testing accuracy of logistic regression...")
    evaluate_log_reg()
