"""Module to extract and load data."""
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = "data/"  # Data directory


def read_dialog_data():
    """Reads data from a path and returns the proper data structure."""

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
    return sentences, dialog_acts


def create_dialog_dataset(test_size=0.15):
    """Creates dataset by reading and splitting dataset."""
    all_x, all_y = read_dialog_data()
    return train_test_split(all_x, all_y, test_size=test_size, random_state=42)


def read_restaurant_dataset():
    return pd.read_csv(os.path.join(DATA_DIR, "restaurant_info.csv"))


def create_augmented_restaurant_dataset():

    FOOD_QUALITY = ["bad", "good", "moderate"]
    CROWDEDNESS = ["quiet", "busy", "moderate"]
    STAY_LENGTH = ["short", "long", "moderate"]
    data = read_restaurant_dataset()
    data["food quality"] = np.random.choice(FOOD_QUALITY, size=len(data))
    data["crowdedness"] = np.random.choice(CROWDEDNESS, size=len(data))
    data["length of stay"] = np.random.choice(STAY_LENGTH, size=len(data))
    return data


def read_augmented_restaurant_dataset():
    return pd.read_csv(os.path.join(DATA_DIR, "restaurant_info_aug.csv"))


if __name__ == "__main__":
    create_augmented_restaurant_dataset().to_csv(
        os.path.join(DATA_DIR, "restaurant_info_aug.csv")
    )
