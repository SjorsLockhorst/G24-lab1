"""Script that allows user to select a model and test it's accuracy on the test set."""
from extract import create_dialog_dataset
from machine_learning import select_model, load_model

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from prettytable import PrettyTable

if __name__ == "__main__":
    # Prompt user to select model
    name, filepath, _ = select_model()
    print(f"Loading {name} model from disk...")

    model = load_model(filepath)

    # Create dataset
    x_train, x_test, y_train, y_test = create_dialog_dataset()

    # Predict values based on test data
    pred = model.predict(x_test)

    def format_percentage(number):
        """Format a number as a percentage"""
        return f"{number * 100:.2f}%"

    # Obtain variables using sklearn metric function
    prec, recall, fscore, n_occurences = precision_recall_fscore_support(
        y_test, pred, zero_division=1
    )

    # Create a table to display information for each label
    table = PrettyTable(["Label", "Precision", "Recall", "F-score", "N occurences"])
    table.add_rows(
        [row for row in zip(model.classes_, prec, recall, fscore, n_occurences)]
    )

    print(f"{name} results:")
    print(table.get_string())

    prec, recall, fscore, n_occurences = precision_recall_fscore_support(
        y_test, pred, zero_division=1, average="weighted"
    )

    print("On average:")
    print(f"Accuracy: {format_percentage(accuracy_score(y_test, pred))}")
    print(f"Precision: {format_percentage(prec)}")
    print(f"Recall: {format_percentage(recall)}")
    print(f"F-score: {format_percentage(fscore)}")

    # print(f"Prediction of {name}:")
    # print(f"Correct: {res * 100:.2f}%")
