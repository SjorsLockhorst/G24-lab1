"""Module to extract and load data into a usuable format."""

import pandas as pd

def read_data():
    """Reads data from a path and returns the proper data structure."""

    f = open('data/dialog_acts.dat', 'r')
    col1 = [] # creating a list for column 1 (dialog act labels)
    col2 = [] # creating a list for column 2 (sentences)
   
    for i in f:
        split_line = i.split(None, 1) #split each line into first word and rest of line. 
        col1.append(split_line[0]) # append first word to list of dialog acts
        col2.append(split_line[1].strip()) # append rest of lines to list of sentences, removing /n 

    columns = {'col1': col1, 'col2': col2}
    df = pd.DataFrame(data=columns) # passing the data into pandas dataframe
    return df


def split_data():
    """Splits data into train and test dataset."""

    #either use numpy random or sklearn traintest split to split the data

    pass


def create_dataset():
    """Creates dataset by reading and splitting dataset."""
    pass

read_data()