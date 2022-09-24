import abc
from typing import Optional

from machine_learning import load_model
from extract import create_restaurant_dataset
from dataclasses import dataclass
from templates import match_area, match_food, match_pricerange, match_request

import numpy as np
import pandas as pd

log_reg = load_model("log_reg.pickle")

data = create_restaurant_dataset()


class StateInterface(metaclass=abc.ABCMeta):
    def __init__(self, number, next_state, end=False):
        self.number = number
        self.end = end
        self.next_state = next_state

    @classmethod
    def __subclasshook__(cls, subclass):
        return hasattr(subclass, "activate") and callable(subclass.activate)

    @abc.abstractmethod
    def activate(self, information, recommendations):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(number={self.number}, end={self.end})"


class WelcomeState(StateInterface):
    def activate(self, information, recommendations):
        sentence = input(
            "Hello , welcome to the Cambridge restaurant system? You can ask for "
            "restaurants by area , price range or food type . How may I help you?\n"
        )
        # Extract information from sentence
        new_information = get_information(sentence)

        new_recommendations = query_information(data, new_information)

        return sentence, new_information, new_recommendations


class ThankYouState(StateInterface):
    def activate(self, information, recomendations):
        print("You're welcome!\n")
        return "", information, recomendations


class ByeState(StateInterface):
    def activate(self, information, recommendations):
        print("Good bye\n")
        return "", information, recommendations


class AskPriceRangeState(StateInterface):
    def activate(self, information, recommendations):
        sentence = ""
        if len(recommendations["pricerange"].unique()) > 1:
            while not information.pricerange:
                sentence = input(
                    "Would you like something in the cheap, moderate or expensive price range?\n"
                )
                information.pricerange = match_pricerange(sentence)

        new_recommendations = query_information(data, information)

        return sentence, information, new_recommendations


class AskTypeState(StateInterface):
    def activate(self, information, recommendations):
        sentence = ""
        if len(recommendations["food"].unique()) > 1:
            while not information.food:
                sentence = input("What kind of food would you like?\n")
                information.food = match_food(sentence)
        new_recommendations = query_information(data, information)
        return sentence, information, new_recommendations


class AskAreaState(StateInterface):
    def activate(self, information, recommendations):
        sentence = ""
        if len(recommendations["area"].unique()) > 1:
            while not information.area:
                sentence = input("What kind of area would you like?\n")
                information.area = match_area(sentence)
        new_recommendations = query_information(data, information)
        return sentence, information, new_recommendations


class RecommendPlaceState(StateInterface):
    def activate(self, information, recommendations):
        try:
            new_recommendations = recommendations.drop(index=len(data))
        except KeyError:
            new_recommendations = recommendations

        if len(new_recommendations) > 0:

            recommendation = new_recommendations.sample()
            new_recommendations = new_recommendations.rename(
                index={recommendation.index[0]: len(data)}
            )

            message = f"{recommendation['restaurantname'].values[0]} is a nice place"
            if information.pricerange:
                message += f" that is {information.pricerange} in price"
            if information.area:
                message += f" in the {information.area} of town"
            if information.food:
                message += f" that serves {information.food} food"
            message += ".\n"
            sentence = input(message)
            new_information = match_request(sentence, information)
            return sentence, new_information, new_recommendations
        return "", information, new_recommendations


class NotFoundState(StateInterface):
    def activate(self, information, recommendations):
        sentence = ""
        message = "There is no (other) restaurant "
        if information.pricerange:
            message += f" that is {information.pricerange} in price"
        if information.area:
            message += f" in the {information.area} of town"
        if information.food:
            message += f" that serves {information.food} food."
        message += "\nPlease try again.\n"
        sentence = input(message)
        information.update(get_information(sentence))
        return sentence, information, data


class RequestInformation(StateInterface):
    def activate(self, information, recommendations):
        recommendation = recommendations.loc[len(data)]
        columns = information.get_requested_columns()
        col_to_str = {
            "addr": "address",
            "phone": "phone number",
            "food": "food",
            "postcode": "postcode",
            "pricerange": "price range",
            "area": "area",
        }
        if columns:
            message = f"Restaurant {recommendation['restaurantname']} "
            i = 0
            while i < len(columns):
                column = columns[i]
                message += f"has {col_to_str[column]}: {recommendation[column]}"
                if i + 1 < len(columns):
                    message += " and "
                i += 1
            message += ".\n"
        else:
            message = "I did not understand your request, please try again.\n"

        sentence = input(message)
        information = match_request(sentence, information)
        return sentence, information, recommendations


def query(data, expected):
    col, value = expected
    return data[data[col] == value]


def query_information(data, information):
    # TODO: Query to take into account when user expresses no preference.
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

    postcode_requested: bool = False
    address_requested: bool = False
    phone_requested: bool = False
    pricerange_requested: bool = False
    area_requested: bool = False
    food_requested: bool = False

    def reset_requests(self):
        self.postcode_requested = False
        self.address_requested = False
        self.phone_requested = False
        self.pricerange_requested = False
        self.area_requested = False
        self.food_requested = False

    def update(self, other):
        if other.pricerange:
            self.pricerange = other.pricerange
        if other.area:
            self.area = other.area
        if other.food:
            self.food = other.food

    def get_requested_columns(self):
        columns = []
        if self.postcode_requested:
            columns.append("postcode")
        if self.address_requested:
            columns.append("addr")
        if self.phone_requested:
            columns.append("phone")
        if self.pricerange_requested:
            columns.append("pricerange")
        if self.area_requested:
            columns.append("area")
        if self.food_requested:
            columns.append("food")
        return columns

    @property
    def complete(self):
        return self.pricerange and self.area and self.food


def get_information(sentence):
    return Information(
        match_pricerange(sentence), match_area(sentence), match_food(sentence)
    )


bye = ByeState(8, None, end=True)

extra_info = RequestInformation(7, None)
recommend = RecommendPlaceState(6, {"bye": bye, "request": extra_info, "thankyou": bye})

not_found = NotFoundState(5, None)
type_food = AskTypeState(4, {"inform": recommend})
ask_area = AskAreaState(3, {"inform": type_food})
price_range = AskPriceRangeState(2, {"inform": ask_area})

not_found.next_state = {"inform": price_range}
recommend.next_state["reqalts"] = recommend
extra_info.next_state = recommend.next_state

welcome = WelcomeState(1, price_range)


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
        if len(new_recommendations) > 0:

            # Only if we have a transition we can make, use this next state
            if isinstance(state.next_state, dict):
                if dialog_act in state.next_state:
                    next_state = state.next_state[dialog_act]
                # Otherwise, repeat this state
                else:
                    next_state = state
            else:
                next_state = state.next_state

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


if __name__ == "__main__":
    # Activate first state
    transition(welcome, verbose=True)
