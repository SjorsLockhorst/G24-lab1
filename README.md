# Lab 1 Introduction to Methods in AI Research

## Installation

To install, create a new virtual environment (using python venv or Conda), activate it and run:

```
pip install -r requirements.txt
```

## Usage

This package consists of some scripts that implement different functionalities.

### Baselines

To evaluate the baseline results, run [baseline.py](baseline.py).

### Machine Learning

We have implemented various Machine Learning classifiers for this assignment.
You can find the code that trains them in [train.py](train.py).
For convenience, we have pretrained them and stored them in [models/](models/) as pickle files.
All other parts of the code that load the models, load these pickle files directly.
To run the interactive CLI environment where you can type sentences, and the system predicts the dialog act based on a selected model, run [predict.py](predict.py).

### Additional

We wrote the [extract.py](extract.py) module as a helper to load in data from the raw data file(s) in [data/](data/).
Additionally, we wrote a bag-of-words vectorizer. This was done to gather understanding on how this works, but in the end, the implementation of sklearn was used.

For the inference part, we need to randomly generate additional columns to the restaurant dataset at random. We did this and saved the csv file, but kept our randomization script.
You could randomize new columns by running [extract.py](extract.py) directly.

### Dialog system

Run [dialog_system.py](dialog_system.py) to test our dialog system.
