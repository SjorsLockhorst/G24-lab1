"""Module to extract and load data into a usuable format."""

import pandas as pd
from sklearn import model_selection as ms

def read_data(datafile):
    """
    Params: file pathway of file to be used.
    Returns: pandas dataframe
    """

    f = open(datafile, 'r')
    col1 = [] # creating a list for column 1 (dialog act labels)
    col2 = [] # creating a list for column 2 (sentences, bags of words)
   
    for i in f:
        split_line = i.split(None, 1) #split each line into first word and rest of line. 
        col1.append(split_line[0]) # append first word to list of dialog acts
        col2.append(split_line[1].lower().strip()) # append rest of lines to list of sentences, removing /n 


    columns = {'labels': col1, 'text': col2}
    df = pd.DataFrame(data=columns) # passing the data into pandas dataframe
    return df


def split_data(unsplit_data):
    """
    params: pd dataframe of data
    returns: train and test sets (as pd dataframes)
    """
    #using sklearn train_test_split to split the dataframe into 85/15 train,test sets. 
    train_set, test_set = ms.train_test_split(unsplit_data, test_size = 0.15, random_state = 3, shuffle=True)
    return train_set, test_set

def create_dataset(datafile): 
    """
    Params = file
    Returns = train_set, test_set in a pd dataframe
    """
    unsplit_data = read_data(datafile) #read_data returns the pandas dataframe
    train_set, test_set = split_data(unsplit_data) #split_data returns the test and train (sub)sets of the pd dataframe
    return train_set, test_set


dialog_acts = 'data/dialog_acts.dat' # if we change the dataset, we don't need to change the functions by defining it here
create_dataset(dialog_acts)