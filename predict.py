from vectorize import vectorize
from machine_learning import load_model


if __name__ == "__main__":
    sent = input("Please enter sentence: ")
    log_reg, vocabulary = load_model()
    x = vectorize(sent.lower(), vocabulary)
    print(log_reg.predict(x.reshape(1, -1)))
