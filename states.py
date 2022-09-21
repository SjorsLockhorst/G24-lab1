import abc
from typing import Optional
from random import randint

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
    def activate(self, information, *args):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(number={self.number}, end={self.end})"


class WelcomeState(StateInterface):
    def activate(self, information, *args):
        return input(
            "Hello , welcome to the Cambridge restaurant system? You can ask for "
            "restaurants by area , price range or food type . How may I help you?\n"
        )


class ThankYouState(StateInterface):
    def activate(self, information, *args):
        print("You're welcome!")


class ByeState(StateInterface):
    def activate(self, information, *args):
        print("Good bye")


class AskPriceRangeState(StateInterface):
    def activate(self, information, *args):
        if not information.pricerange:
            return input(
                "Would you like something in the cheap, moderate or expensive price range?"
            )
        return ""


class AskTypeState(StateInterface):
    def activate(self, information, *args):
        if not information.food:
            return input("What kind of food would you like?\n")
        return ""


class RecommendPlaceState(StateInterface):
    def activate(self, information, *args):
        recommendations = args[0]

        recommendation = recommendations.sample()
        recommendations = recommendations.drop(index=recommendation.index[0])

        message = f"{recommendation['restaurantname'].values[0]} is a nice place"
        if information.pricerange:
            message += f" that is {information.pricerange} in price"
        if information.area:
            message += f" in the {information.area} of town"
        if information.food:
            message += f" that serves {information.food} food"
        message += ".\n"
        return input(message)


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
    return Information("cheap", None, "italian")


def transition(
    state: StateInterface,
    data: pd.DataFrame,
    information: Information = Information(None, None, None),
    recommendations: pd.DataFrame = pd.DataFrame({}),
    model=log_reg,
    verbose=False,
):
    # First of all, activate the state
    sentence = state.activate(information, recommendations)

    # While we are not at the end
    if not state.end:

        # If state returned a sentence
        if sentence:
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

        # If the state returned nothing, interpret it as a skip
        else:
            dialog_act = "skip"
            updated_information = information
            new_recommendations = recommendations
            next_state = state.next_state_map[dialog_act]

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
    bye = ByeState(4, None, end=True)

    recommend = RecommendPlaceState(4, {"bye", bye})
    type_food = AskTypeState(3, {"inform": recommend, "skip": recommend})
    price_range = AskPriceRangeState(2, {"inform": type_food, "skip": type_food})
    welcome = WelcomeState(1, {"inform": price_range})

    transition(welcome, data, verbose=True)
