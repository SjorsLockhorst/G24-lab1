import abc
from typing import Optional

from machine_learning import load_model
from extract import create_restaurant_dataset
from dataclasses import dataclass

import numpy as np
import pandas as pd

log_reg = load_model("log_reg.pickle")


class StateInterface(metaclass=abc.ABCMeta):
    def __init__(self, number, next_state_map, end=False):
        self.number = number
        self.end = end
        self.next_state_map = next_state_map

    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "activate") and callable(subclass.activate)

    @abc.abstractmethod
    def activate(self):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(number={self.number}, end={self.end})"


class WelcomeState(StateInterface):
    def __init__(self, number, next_state_map):
        super().__init__(number, next_state_map)

    def activate(self):
        print("Welcome user")


class ThankYouState(StateInterface):
    def __init__(self, number, next_state_map):
        super().__init__(number, next_state_map)

    def activate(self):
        print("You're welcome!")


class ByeState(StateInterface):
    def __init__(self, number, next_state_map):
        super().__init__(number, next_state_map, end=True)

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

    def update(self, other: "Information"):
        if other.pricerange:
            self.pricerange = other.pricerange
        if other.area:
            self.area = other.area
        if other.food:
            self.food = other.food
        return self

    @property
    def complete(self):
        return self.pricerange and self.area and self.food


def get_information(sentence):
    # TODO: Add actual extraction of information
    return Information("cheap", "south", None)


def transition(
    state: StateInterface,
    data: pd.DataFrame,
    information: Information = Information(None, None, None),
    recommendations: pd.DataFrame = pd.DataFrame({}),
    model=log_reg,
    verbose=False,
):
    # First of all, activate the state
    state.activate()

    # While we are not at the end
    if not state.end:

        # Get user input
        sentence = input()

        # User given model to predict dialog act
        dialog_act = model.predict([sentence.lower()])[0]

        # Extract information from sentence
        new_information = get_information(sentence)

        # Update information with current information
        updated_information = information.update(new_information)

        # Query data and update recommendations, based on new information
        new_recommendations = query_information(data, new_information)

        # Only if we have a transition we can make, use this next state
        if dialog_act in state.next_state_map:
            next_state = state.next_state_map[dialog_act]

        # Otherwise, repeat this state
        else:
            next_state = state
        if verbose:
            print(f"Dialog act: {dialog_act}")
            print(f"Previous state: {state}")
            print(f"Next state: {next_state}")
            print(f"Current information: {updated_information}")
            print(f"Recommended based on information: {new_recommendations}")

        # Recursively call this function with updated information
        transition(
            next_state,
            data,
            information=updated_information,
            recommendations=new_recommendations,
            model=model,
            verbose=verbose,
        )
    else:
        return


if __name__ == "__main__":
    # Activate first state
    data = create_restaurant_dataset()

    bye = ByeState(3, {})
    thank_you = ThankYouState(2, {"bye": bye})
    welcome = WelcomeState(1, {"bye": bye, "thankyou": thank_you})
    transition(welcome, data, verbose=True)
