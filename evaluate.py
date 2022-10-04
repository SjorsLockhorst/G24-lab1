"""Script that allows user to select a model and test it's accuracy on the test set."""
from extract import create_dialog_dataset
from machine_learning import select_model, load_model
from sklearn.metrics import precision_recall_fscore_support

if __name__ == "__main__":
    name, filepath, _ = select_model()
    print(f"Loading {name} model from disk...")
    model = load_model(filepath)
    _, x_test, _, y_test = create_dialog_dataset()
    res = model.predict(x_test)
    prec, recall, fscore, n_occurences = precision_recall_fscore_support(
        y_test, res, zero_division=1
    )

    def format_percentage(number):
        return f"{number * 100:.2f}%"

    def print_metric(metric, metric_name, percentage=True):
        print(f"{metric_name}:\n")
        for class_, metric in zip(model.classes_, metric):
            if percentage:
                print(f"{class_}: {format_percentage(metric)}")
            else:
                print(f"{class_}: {metric}")
        print()

    print_metric(prec, "Precision")
    print_metric(recall, "Recall")
    print_metric(fscore, "F-score")
    print_metric(n_occurences, "N per category", percentage=False)

    prec, recall, fscore, n_occurences = precision_recall_fscore_support(
        y_test, res, zero_division=1, average="macro"
    )

    print("On average:")
    print(f"Precision: {format_percentage(prec)}")
    print(f"Recall: {format_percentage(recall)}")
    print(f"F-score: {format_percentage(fscore)}")

    # print(f"Prediction of {name}:")
    # print(f"Correct: {res * 100:.2f}%")
