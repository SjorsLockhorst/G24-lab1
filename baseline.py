"""Module that implements the two baselines as mentioned in the assignments."""
from collections import Counter
import re

import numpy as np

from extract import read_data


def get_most_frequent(y):
    """Get the most frequent value in array y."""
    return Counter(y).most_common(1)[0][0]


def assign_most_frequent(y):
    """Create an array that contains most frequent values of y only."""
    return [get_most_frequent(y)] * len(y)


def assign_rule_based(x, most_frequent="inform"):
    """
    Assign a dialog act to each sentence in x based on matching patterns.

    If no pattern matches, predict most frequent.
    """
    pattern_mapping = [
        (r"^.*thank.*$", "thankyou"),
        (r"^.*(what|address|phone|number|post? code|zip? code).*$", "request"),
        (r"^.*(what|how) about.*$", "reqalts"),
        (r"^.*(yes|correct).*$", "affirm"),
        (r"^.*(cheap|price|expensive).*$", "inform"),
        (r"^.*(\wnot\w|\wno\w).*$", "negate"),
        (r"^.*(good? bye)|(\wbye\w).*$", "bye"),
        (r"^.*(unintellgible|noisy|cough|tv_noise).*$", "null"),
        (r"^.*(hello).*$", "hello"),
    ]
    # Initialise an empty list to save predictions in
    predicted_labels = []

    # Loop over each sentence in the data
    for sentence in x:

        # Initialise variables for the loop
        i = 0
        assigned = False

        # While there are patterns left to try, and nothing has yet been assigned
        while not assigned and i < len(pattern_mapping):

            # Get the predicate and label that belongs to it from the pattern mapping
            patt, label = pattern_mapping[i]

            # If our pattern matches
            if re.match(patt, sentence):

                # Set assigned, so we won't loop again
                assigned = True

                # Add label to the predictions
                predicted_labels.append(label)
            i += 1
        # If after looping we didn't assign any label, just assign the most frequent
        if not assigned:
            predicted_labels.append(most_frequent)
    return predicted_labels  # Finally, return all our predictions


def evaluate(labels, predictions):
    """Returns what percentage of predictions matches the given labels."""

    # Make numpy arrays to get boolean array, for each element True if they match.
    matches = np.array(labels) == np.array(predictions)

    # Get the total number of matches, and calculate percentage
    n_correct = np.sum(matches)
    percentage_correct = n_correct / len(matches) * 100
    return percentage_correct


if __name__ == "__main__":
    print("Loading raw data...")
    xs, ys = read_data()
    most_frequent = get_most_frequent(ys)

    print("Baseline 1: Assigning most frequent..")
    most_frequent_ys = assign_most_frequent(ys)
    print()
    print(f"Most frequent {most_frequent}")
    print(f"Correct: {evaluate(ys, most_frequent):.2f}%")
    print()

    print("Baseline 2: Assigning labels based on rules...")
    y_pred = assign_rule_based(xs, most_frequent=most_frequent)
    print()
    print(f"Correct {evaluate(ys, y_pred):.2f}%")
