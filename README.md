# Lab 1 Introduction to Methods in AI Research

## Installation

To install run:

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
For convenience, we have pretrained them and store them in [models/](models/) as pickled files.
All other parts of the code that load the models, load these pickle files directly.

To evaluate the different machine learning algorithms implemented, run [evaluate.py](evaluate.py).

To run the interactive CLI environment where you can type sentences, and the system predicts the dialog act based on a selected model, run [predict.py](predict.py)

### Additional

We wrote the [extract.py](extract.py) module as a helper to load in data from the raw data file(s) in [data/](data/).
Additionally, we wrote a bag-of-words vectorizer. This was done to gather understanding on how this works, but in the end, the implementation of sklearn was used.
We kept our original in this repository for reference in the file [vectorize.py](vectorize.py).
