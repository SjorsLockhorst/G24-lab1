import abc
from re import match
from typing import Optional
from random import randint

from machine_learning import load_model
from extract import create_restaurant_dataset
from dataclasses import dataclass
from templates import match_area, match_food, match_pricerange

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
    def activate(self, information, recommendations):
        data = create_restaurant_dataset()
        sentence = input(
            "Hello , welcome to the Cambridge restaurant system? You can ask for "
            "restaurants by area , price range or food type . How may I help you?\n"
        )
        # Extract information from sentence
        new_information = get_information(sentence)

        return sentence, new_information, data


class ThankYouState(StateInterface):
    def activate(self, information, recomendations):
        print("You're welcome!\n")
        return None, information, recomendations


class ByeState(StateInterface):
    def activate(self, information, recommendations):
        print("Good bye\n")
        return None, information, recommendations


class AskPriceRangeState(StateInterface):
    def activate(self, information, recommendations):
        sentence = ""
        if len(recommendations["pricerange"].unique()) > 1:
            while not information.pricerange:
                sentence = input(
                    "Would you like something in the cheap, moderate or expensive price range?\n"
                )
                information.pricerange = match_pricerange(sentence)

        return sentence, information, recommendations


class AskTypeState(StateInterface):
    def activate(self, information, recommendations):
        sentence = ""
        if len(recommendations["food"].unique()) > 1:
            while not information.food:
                sentence = input("What kind of food would you like?\n")
                information.food = match_food(sentence)
        return sentence, information, recommendations


class AskAreaState(StateInterface):
    def activate(self, information, recommendations):
        sentence = ""
        if len(recommendations["area"].unique()) > 1:
            while not information.area:
                sentence = input("What kind of area would you like?\n")
                information.area = match_area(sentence)
        return sentence, information, recommendations


class RecommendPlaceState(StateInterface):
    def activate(self, information, recommendations):
        recommendation = recommendations.sample()

        message = f"{recommendation['restaurantname'].values[0]} is a nice place"
        if information.pricerange:
            message += f" that is {information.pricerange} in price"
        if information.area:
            message += f" in the {information.area} of town"
        if information.food:
            message += f" that serves {information.food} food"
        message += ".\n"
        return input(message), information, recommendations


class NotFoundState(StateInterface):
    def activate(self, information, recommendations):
        sentence = ""
        message = "There is no restaurant "
        if information.pricerange:
            message += f" that is {information.pricerange} in price"
        if information.area:
            message += f" in the {information.area} of town"
        if information.food:
            message += f" that serves {information.food} food."
        message += "\nPlease try again.\n"
        sentence = input(message)
        information.update(get_information(sentence))

        return sentence, information, recommendations


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

    def update(self, other):
        if other.pricerange:
            self.pricerange = other.pricerange
        if other.area:
            self.area = other.area
        if other.food:
            self.food = other.food

    @property
    def complete(self):
        return self.pricerange and self.area and self.food


def get_information(sentence):
    return Information(
        match_pricerange(sentence), match_area(sentence), match_food(sentence)
    )


bye = ByeState(4, None, end=True)

recommend = RecommendPlaceState(5, {"bye": bye})
ask_area = AskAreaState(4, {"inform": recommend})
type_food = AskTypeState(3, {"inform": ask_area})
price_range = AskPriceRangeState(2, {"inform": type_food})
not_found = NotFoundState(6, {"inform": price_range})
welcome = WelcomeState(1, {"inform": price_range})


def transition(
    state: StateInterface,
    information: Information = Information(None, None, None),
    recommendations: pd.DataFrame = pd.DataFrame({}),
    model=log_reg,
    verbose=False,
):
    # First of all, activate the state
    sentence, updated_information, new_recommendations = state.activate(
        information, recommendations
    )

    # While we are not at the end
    if not state.end:

        # User given model to predict dialog act
        dialog_act = model.predict([sentence.lower()])[0]

        # Query data and update recommendations, based on new information
        # TODO: Fix what happens when 0 rows appear
        # TODO: Go to request when 1 row remains only
        if len(new_recommendations) > 0:
            new_recommendations = query_information(
                new_recommendations, updated_information
            )

            # Only if we have a transition we can make, use this next state
            if dialog_act in state.next_state_map:
                next_state = state.next_state_map[dialog_act]

            # Otherwise, repeat this state
            else:
                next_state = state

        else:
            next_state = not_found

        if verbose:
            print(f"Dialog act: {dialog_act}")
            print(f"Previous state: {state}")
            print(f"Next state: {next_state}")
            print(f"Current information: {updated_information}")
            print(f"Recommended based on information: {new_recommendations}")

        # Recursively call this function with updated information
        transition(
            next_state,
            information=updated_information,
            recommendations=new_recommendations,
            model=model,
            verbose=verbose,
        )
    else:
        return


if __name__ == "__main__":
    # Activate first state

    transition(welcome, verbose=True)
