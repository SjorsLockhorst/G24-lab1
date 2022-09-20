import abc
from typing import Optional

from machine_learning import load_model
from extract import create_restaurant_dataset
from dataclasses import dataclass

import numpy as np
import pandas as pd

log_reg = load_model("log_reg.pickle")


class StateInterface(metaclass=abc.ABCMeta):
    def __init__(self, number, end=False):
        self.number = number
        self.end = end

    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "activate") and callable(subclass.activate)

    @abc.abstractmethod
    def activate(self):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(number={self.number}, end={self.end})"


class WelcomeState(StateInterface):
    def __init__(self, number):
        super().__init__(number)

    def activate(self):
        print("Welcome user")


class ThankYouState(StateInterface):
    def __init__(self, number):
        super().__init__(number)

    def activate(self):
        print("Thanks!")


class ByeState(StateInterface):
    def __init__(self, number):
        super().__init__(number)

    def activate(self):
        print("Good bye")


def query(data, expected):
    col, value = expected
    return data[data[col] == value]


def query_information(data, information):
    data = data.copy()
    if information.pricerange:
        data = query(data, ("pricerange", information.pricerange))
    if information.area:
        data = query(data, ("area", information.area))
    if information.food:
        data = query(data, ("food", information.food))
    return data


def entropy(arr):
    p_values = arr.value_counts() / len(arr)
    return -np.sum((np.log(p_values) / np.log(2)) * p_values)


@dataclass
class Information:
    pricerange: Optional[str]
    area: Optional[str]
    food: Optional[str]


def transition(
    state: StateInterface,
    sentence: str,
    information: Information,
    model=log_reg,
    verbose=False,
):
    dialog_act = model.predict([sentence.lower()])[0]
    if verbose:
        print(f"Dialog act: {dialog_act}")
        print(f"Previous state: {state}")
        print("Next state: 'TODO: put next state'")
    # Extract any information from user input
    # Fill slots, pricerange, area, food

    return state, information


if __name__ == "__main__":
    # Activate first state
    data = create_restaurant_dataset()
    print(query_information(data, Information("cheap", "west", None)))
    # bye = ByeState(1)
    # welcome = WelcomeState(1)

    # current_state: StateInterface = welcome
    # current_information = Information(None, None, None)
    # VERBOSE = True
    # while not current_state.end:
    #     current_state.activate()
    #     user_input = input()
    #     current_state, current_information = transition(
    #         current_state, user_input, current_information, verbose=VERBOSE
    #     )
    # current_state.activate()
