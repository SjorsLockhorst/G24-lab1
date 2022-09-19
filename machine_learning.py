"""Module that implements different ML classifiers and some utility functions."""
import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


# Models that are implemented
MODELS = [
    ("Logistic regression", "log_reg.pickle", LogisticRegression),
    ("Multinomial naive bayes", "multi_nb.pickle", MultinomialNB),
    ("Random forest classifier", "random_forest.pickle", RandomForestClassifier),
    ("Descision tree classifier", "descision_tree.pickle", DecisionTreeClassifier),
    ("K-nearest neighbors classifier", "k_nearest.pickle", KNeighborsClassifier),
]

# Directory where pickle files are saved
MODEL_DIR = "models/"


def load_model(filename):
    """Load model from pickle file in models directory."""

    with open(os.path.join(MODEL_DIR, filename), "rb") as file:
        return pickle.load(file)


def train_model(train_x, train_y, classifier_model):
    """Train a model."""
    pipe = Pipeline(
        [("vectorizer", CountVectorizer()), ("classifier", classifier_model())]
    )
    return pipe.fit(train_x, train_y)


def save_model(model, filename):
    """Save a model to pickle file in models directory."""
    with open(os.path.join(MODEL_DIR, filename), "wb") as f:
        pickle.dump(model, f)


def select_model():
    """CLI interface to let user select one of the ML classifiers."""
    print("Please select a model")
    selected = 100
    while selected > len(MODELS) - 1 or selected < 0:
        for idx, (name, _, _) in enumerate(MODELS, start=1):
            print(f"[{idx}]: {name}")
        try:
            selected = int(input("\n")) - 1
        except ValueError:
            print(f"Please select a value in {list(range(1, len(MODELS) + 1))}")
    return MODELS[selected]
