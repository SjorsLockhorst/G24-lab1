"""Script that allows user to select a model and test it's on some stats."""
import os

from extract import create_dialog_dataset
from machine_learning import select_model, load_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    ConfusionMatrixDisplay,
)
from prettytable import PrettyTable

if __name__ == "__main__":
    # Prompt user to select model
    model_name, filepath, _ = select_model()
    print(f"Loading {model_name} model from disk...")

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

    # Print result sin a table
    print(f"{model_name} results:")
    print(table.get_string())

    # Init confusion matrix
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    fig.set_dpi(100)

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        pred,
        labels=model.classes_,
        cmap="gray",
        xticks_rotation="vertical",
        ax=ax,
    )

    # Retrieve average stats for entire set, weighted by n occurences
    prec, recall, fscore, n_occurences = precision_recall_fscore_support(
        y_test, pred, zero_division=1, average="weighted"
    )

    # Print out stats
    print("On average:")
    print(f"Accuracy: {format_percentage(accuracy_score(y_test, pred))}")
    print(f"Precision: {format_percentage(prec)}")
    print(f"Recall: {format_percentage(recall)}")
    print(f"F-score: {format_percentage(fscore)}")

    # Write out confusion matrix to png file
    PLOT_DIR = "plots"
    plot_path = os.path.join(PLOT_DIR, f"{model_name}_confusion_matrix.png")
    fig.savefig(
        plot_path,
        bbox_inches="tight",
        dpi=300,
    )
    print(f"Saved confusion matrix to {plot_path}.")

    # Create csv file with results, to use for analysis
    df = pd.DataFrame()
    all_x = np.concatenate([x_train, x_test])
    all_y = np.concatenate([y_train, y_test])
    all_pred = model.predict(all_x)
    df["sentence"] = all_x
    df["correct label"] = all_y
    df["predicted label"] = all_pred
    df["is correct"] = all_y == all_pred
    df["is train"] = df.index < len(x_train)

    RESULTS_DIR = "results"
    results_path = os.path.join(RESULTS_DIR, f"{model_name}_results.csv")
    df.to_csv(results_path)
    print(f"Saved confusion matrix to {results_path}.")
