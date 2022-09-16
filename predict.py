import os
import pickle

from vectorize import vectorize


def load_model():
    DIR = "models/"
    LOG_REG_PATH = os.path.join(DIR, "log_reg.pickle")
    VOC_PATH = os.path.join(DIR, "vocabulary.pickle")
    with open(LOG_REG_PATH, "rb") as file:
        model = pickle.load(file)
    with open(VOC_PATH, "rb") as file:
        vocabulary = pickle.load(file)
    return model, vocabulary


if __name__ == "__main__":
    sent = input("Please enter sentence: ")
    log_reg, vocabulary = load_model()
    x = vectorize(sent.lower(), vocabulary)
    print(log_reg.predict(x.reshape(1, -1)))
