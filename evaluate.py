"""Script that allows user to select a model and test it's accuracy on the test set."""
from extract import create_dialog_dataset
from machine_learning import select_model, load_model

if __name__ == "__main__":
    name, filepath, _ = select_model()
    print(f"Loading {name} model from disk...")
    model = load_model(filepath)
    _, x_test, _, y_test = create_dialog_dataset()
    res = model.score(x_test, y_test)
    print(f"Prediction of {name}:")
    print(f"Correct: {res * 100:.2f}%")
