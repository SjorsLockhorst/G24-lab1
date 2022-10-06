"""Script that trains a selected machine learning model."""
import pickle
import os


from extract import create_dialog_dataset
from machine_learning import select_model, train_model, save_model


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = create_dialog_dataset()
    _, filename, model = select_model()
    print("Training model...")
    trained_model = train_model(x_train, y_train, model)
    save_model(trained_model, filename)
