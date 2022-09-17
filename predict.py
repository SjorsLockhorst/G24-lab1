from machine_learning import load_model
from machine_learning import select_model
from baseline import get_most_frequent, assign_rule_based
from extract import create_dataset


def predict_machine_learning(model):
    return model.predict(sentence)


def predict_rule_based(sentence, y):
    return assign_rule_based([sentence])[0]


def predict_most_frequent(sentence, y):
    return get_most_frequent(y)


if __name__ == "__main__":
    y_train = create_dataset()[2]
    sentence = ""
    options = {
        1: predict_most_frequent,
        2: predict_rule_based,
        3: predict_machine_learning,
    }
    print("Which model would you like to test?")
    selected = 100
    while selected not in options:
        selected_str = input(
            "[1]: Assign most frequent label, [2] Assign rule based, [3] Assing using machine learning.\n"
        )
        try:
            selected = int(selected_str)
        except ValueError:
            print("Please enter either 1, 2 or 3.")

    predictor = options[selected]

    if selected == 3:
        _, path, _ = select_model()
        model = load_model(path)

        while sentence != "stop":
            sentence = input("Please enter sentence: (type stop to exit): ")
            print("\n", model.predict([sentence])[0], "\n")

    else:
        while sentence != "stop":

            sentence = input("Please enter sentence: (type stop to exit): ")
            print("\n", predictor(sentence, y_train), "\n")
