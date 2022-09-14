"""Module to extract and load data into a usuable format."""
import os
import math


def read_data():
    """Reads data from a path and returns the proper data structure."""

    DATA_DIR = "data/"  # Data director
    filepath = os.path.join(
        DATA_DIR, "dialog_acts.dat"
    )  # join filename and path to obtain full system path
    dialog_acts = []  # List to store all dialog_acts
    sentences = []  # List to store all sentences

    with open(filepath, "r") as file:  # Open file
        lines = file.readlines()
        for line in lines:
            stripped = line.strip()
            split = stripped.split(" ")
            dialog_act = split[0]
            sentence = " ".join(split[1:])

            # Add first word to list as dialog act
            dialog_acts.append(dialog_act)

            # Add the rest of the line as a string to the sentences
            sentences.append(sentence)
    return dialog_acts, sentences


def split_data(x, y, train_percentage):
    """Splits data into train and test dataset."""
    split = math.floor(len(x) * train_percentage / 100)
    x_train = x[:split]
    y_train = y[:split]
    x_test = x[split:]
    y_test = y[split:]
    return x_train, y_train, x_test, y_test


def create_dataset(train_percentage=85):
    """Creates dataset by reading and splitting dataset."""
    all_x, all_y = read_data()
    return split_data(all_x, all_y, train_percentage)
