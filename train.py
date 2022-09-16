import pickle
import os

from sklearn.linear_model import LogisticRegression

from extract import create_dataset
from vectorize import create_bag_of_words

if __name__ == "__main__":
    DIR = "models/"
    MODEL_FILE = "log_reg.pickle"
    VOC_FILE = "vocabulary.pickle"

    x_train, y_train, x_test, y_test = create_dataset()
    features, vocabulary = create_bag_of_words(x_train)
    model = LogisticRegression(verbose=1).fit(features.T, y_train)
    with open(os.path.join(DIR, MODEL_FILE), "wb") as model_file:
        pickle.dump(model, model_file)
    with open(os.path.join(DIR, VOC_FILE), "wb") as voc_file:
        pickle.dump(vocabulary, voc_file)
    print(f"Done training, saved in {DIR}.")
