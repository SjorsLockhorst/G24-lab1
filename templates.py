import pandas as pd
import re

df = pd.read_csv('data/restaurant_info.csv')

# Making arrays of all possible labels per slot from the dataset
pricerange_labels = df['pricerange'].unique()
area_labels = df['area'].unique()
food_labels = df['food'].unique()

"""Funtions that takes a user's turn (type string) as input after the relevant question.
Checks whether it contains any information that can be used to fill slots. 
If so, assigns that slot. If no type is recognized, should ask the user again."""

def find_pricerange(user_input):
    user_input = user_input.strip.lower()

    SlotFilled = False # First check whether a user explicitly mentions a label
    for word in user_input: 
        if word in pricerange_labels:
            # fill pricerange slot with 'word'
            SlotFilled = True
    
    if not SlotFilled:
        PriceTemplate = re.compile(r"\b\w+\s(priced|pricing|price ?range)\b")
        mo = regex.search(user_input)
            if mo:
                # fill pricerange slot with "mo.group().split()[0]"
            SlotFilled = True
            
    if not SlotFilled:
        # ask user again.

    return

def find_area(user_input):
    user_input = user_input.strip.lower()
    
    SlotFilled = False # First check whether a user explicitly mentions a label
    for word in user_input:
        if word in area_labels:
            # fill area slot with 'word'
            SlotFilled = True

    if not SlotFilled:
        AreaTemplate = re.compile(r"\b\w+\s(part)\b")
        mo = regex.search(user_input)
            if mo:
                # fill pricerange slot with "mo.group().split()[0]"
            SlotFilled = True

    if not SlotFilled:
        AreaTemplate2 = re.compile(r"\b(in the|somewhere)\s\w+\b")
        mo = regex.search(user_input)
            if mo:
                # fill pricerange slot with "mo.group().split()[-1]"
            SlotFilled = True
            
    if not SlotFilled:
        # ask user again.

    return


def find_food(user_input):
    user_input = user_input.strip.lower()
    
    SlotFilled = False # First check whether a user explicitly mentions a label
    for word in user_input:
        if word in food_labels:
            # fill food slot with 'word'
            SlotFilled = True

    if not SlotFilled:
        FoodTemplate = re.compile(r"\b\w+\s(food|cuisine|kitchen|restaurant|place)\b")
        mo = regex.search(user_input)
            if mo:
                # fill pricerange slot with "mo.group().split()[0]"
            SlotFilled = True
            
    if not SlotFilled:
        # ask user again.

    return