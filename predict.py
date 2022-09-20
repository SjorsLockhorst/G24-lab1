"""Module for predicting the dialog act of a user input sentence."""
import sys

from machine_learning import load_model
from machine_learning import select_model
from baseline import get_most_frequent, assign_rule_based
from extract import create_dialog_dataset


def predict_machine_learning(model):
    """Simple wrapper function to predict dialog act using a ML model."""
    return model.predict(sentence)


def predict_rule_based(sentence, y):
    """Simple wrapper function to predict dialog act using rule based function."""
    return assign_rule_based([sentence])[0]


def predict_most_frequent(sentence, y):
    """Simple wrapper function to predict dialog act by assigning most frequent label"""
    return get_most_frequent(y)


if __name__ == "__main__":
    # Obtain training data
    y_train = create_dialog_dataset()[2]
    sentence = ""
    selected = sys.maxsize * 2 + 1

    # Initialise options
    options = {
        1: predict_most_frequent,
        2: predict_rule_based,
        3: predict_machine_learning,
    }
    print("Which model would you like to test?")
    # Loop while user has not selected a valid option
    while selected not in options:
        # Get user input string
        selected_str = input(
            "[1]: Assign most frequent label\n[2]: Assign rule based\n"
            "[3]: Assign using machine learning\n"
        )

        # Attempt to cast to integer
        try:
            selected = int(selected_str)

        # If this fails, user has not entered a valid integer, notify them.
        except ValueError:
            print("Please enter a valid integer within range.")

    print()
    # Once a valid option was picked, select the predictor
    predictor = options[selected]

    # If it was an ml model, load it
    if selected == 3:
        _, path, _ = select_model()
        model = load_model(path)

        # Loop while user doesn't type stop
        while sentence != "stop":
            # Ask for input sentence
            sentence = input("Please enter sentence: (type stop to exit): ")
            # Predict based on input sentence
            print("\nDialog act: ", model.predict([sentence.lower()])[0], "\n")

    else:
        while sentence != "stop":

            # Ask for input sentence
            sentence = input("Please enter sentence: (type stop to exit): ")
            # Predict based on input sentence
            print("\nDialog act: ", predictor(sentence.lower(), y_train), "\n")
