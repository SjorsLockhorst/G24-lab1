from vectorize import vectorize
from machine_learning import load_model
from baseline import get_most_frequent, assign_rule_based
from extract import create_dataset


def predict_log_reg(sentence, y):
    log_reg, vocabulary = load_model()
    x = vectorize(sentence.lower(), vocabulary)
    return log_reg.predict(x.reshape(1, -1))[0]


def predict_rule_based(sentence, y):
    return assign_rule_based([sentence])[0]


def predict_most_frequent(sentence, y):
    return get_most_frequent(y)


if __name__ == "__main__":
    y_train = create_dataset()[2]
    sentence = ""
    options = {1: predict_most_frequent, 2: predict_rule_based, 3: predict_log_reg}
    print("Which model would you like to test?")
    selected = 100
    while selected not in options:
        selected_str = input(
            "[1]: Assign most frequent label, [2] Assign rule based, [3] Assign using logistic regression model.\n"
        )
        try:
            selected = int(selected_str)
        except ValueError:
            print("Please enter either 1, 2 or 3.")

    predictor = options[selected]

    while sentence != "stop":
        sentence = input("Please enter sentence: (type stop to exit): ")
        print("\n", predictor(sentence, y_train), "\n")
