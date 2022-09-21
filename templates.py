"""
Writing the templates that will extract the user's intention. These intentions will be used to fill the user
slots that we need to fill (categories from the dataset):
    pricerange
    area
    food

Essentially keyword matching. We want to know if any part of the userinput can fill a slot.*
Assign keywords to each label present in the dataset. Eg. if userinput contains 'cheap', assign pricerange = cheap

* If a user fills more than one slot in one turn, we should probably implement an exception that skips the user
questions that now become irrelevant. 
"""

import pandas as pd

df = pd.read_csv('data/restaurant_info.csv')

# Making arrays of all possible labels per slot from the dataset
pricerange_labels = df['pricerange'].unique()
area_labels = df['area'].unique()
food_labels = df['food'].unique()

def fill_slots(user_input):
    """
    Takes a user's turn (type string) as input
    checks whether it contains any information that can be used to fill slots immediately. 
    If so, assign the slot (eg pricerange) with the specified label (eg cheap).

    If the string doesn't contain a label exactly, look into using templates
    """

    user_input = user_input.strip.lower()

    SlotFilled = False

    # First check whether a user explicitly mentions a label
    for word in user_input:
        if word in pricerange_labels:
            # set pricerange label as 'word'
            SlotFilled = True

    for word in user_input:
        if word in area_labels:
            # set area slot as 'word'
            SlotFilled = True


    for word in user_input:
        if word in food_labels:
            # set food slot as 'word'
            SlotFilled = True

    # If a user doesn't fill a label, move on to checking if any content was similar to the labels.

    if not SlotFilled:
        # Move on to templates that we prespecified, in the form of:
        # if "{variable} food" in userinput, check if variable in food_labels
        # if variable matches a food_label, assign that slot with that label.

        pass