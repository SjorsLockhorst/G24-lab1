import abc
import math
from typing import Optional

from machine_learning import load_model
from extract import read_augmented_restaurant_dataset
from dataclasses import dataclass
from templates import (
    match_area,
    match_food,
    match_pricerange,
    match_request,
    match_food,
    match_consequent,
)

import numpy as np
import pandas as pd

log_reg = load_model("log_reg.pickle")

data = read_augmented_restaurant_dataset()


class StateInterface(metaclass=abc.ABCMeta):
    """Interface that dictates any state must have a activate function."""

    def __init__(self, number, next_state, end=False):
        """Universal init method for all state instance."""
        self.number = number
        self.end = end
        self.next_state = next_state

    @classmethod
    def __subclasshook__(cls, subclass):
        """Classes that have a callable activate property be a valid StateInterface."""
        return hasattr(subclass, "activate") and callable(subclass.activate)

    @abc.abstractmethod
    def activate(self, information, recommendations):
        """Abstract method that subclasses must implement."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(number={self.number}, end={self.end})"


@dataclass
class Inference:
    consequent: str
    truth_value: bool
    verb: str
    because: str

    pricerange: Optional[str] = None
    food_quality: Optional[str] = None
    length_of_stay: Optional[str] = None
    food_type: Optional[str] = None
    crowdedness: Optional[str] = None

    def infer_from_data(self, recommendations):
        new_rec = recommendations.copy()
        if self.pricerange:
            new_rec = new_rec[new_rec["pricerange"] == self.pricerange]
        if self.food_type:
            new_rec = new_rec[new_rec["food"] == self.food_type]
        if self.food_quality:
            new_rec = new_rec[new_rec["food quality"] == self.food_quality]
        if self.length_of_stay:
            new_rec = new_rec[new_rec["length of stay"] == self.length_of_stay]
        if self.crowdedness:
            new_rec = new_rec[new_rec["crowdedness"] == self.crowdedness]

        return new_rec

    @property
    def consequent_sent(self):
        return f"{self.verb} {self.consequent}"

    def __str__(self):
        return f"The restaurant {self.consequent_sent} because {self.because}.\n"


class Inferences:
    def __init__(self, consequent, if_true=None, if_false=None):
        self.consequent = consequent
        self.if_true = if_true
        self.if_false = if_false
        self.truth_value = None
        if not if_true and not if_false:
            raise ValueError("Must have at least one inference")

    def infer(self, recommendations, truth_value):
        new_rec = recommendations.copy()
        self.truth_value = truth_value
        if truth_value:
            if self.if_true is not None and self.if_false is not None:
                new_rec = self.if_true.infer_from_data(new_rec)
                to_remove = self.if_false.infer_from_data(new_rec)
                new_rec = new_rec.drop(index=to_remove.index)
            elif self.if_true is not None and self.if_false is None:
                new_rec = self.if_true.infer_from_data(new_rec)
            elif self.if_true is None and self.if_false is not None:
                new_rec = pd.DataFrame()

        else:
            if self.if_false is not None and self.if_true is not None:
                new_rec = self.if_false.infer_from_data(new_rec)
                to_remove = self.if_true.infer_from_data(new_rec)
                new_rec = new_rec.drop(index=to_remove.index)
            elif self.if_false is not None and self.if_false is None:
                new_rec = self.if_false.infer_from_data(new_rec)
            elif self.if_false is None and self.if_true is not None:
                new_rec = pd.DataFrame()

        return new_rec

    @property
    def chosen_inference(self):
        if self.truth_value is None:
            return None
        if self.truth_value:
            return self.if_true
        else:
            return self.if_false

    @property
    def consequent_sent(self):
        # TODO: Make this a bit less ugly, add actual formatted text
        if self.chosen_inference is None:
            return f"has value {self.truth_value} for {self.consequent}"
        return self.chosen_inference.consequent_sent

    @property
    def message(self):
        if self.chosen_inference is not None:
            return str(self.chosen_inference)
        else:
            return f"has value {self.truth_value} for {self.consequent}"


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
        information.inferences = None

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
                message += f" that is {recommendation['pricerange'].values[0]} in price"
            if information.area:
                message += f" in the {recommendation['area'].values[0]} of town"
            if information.food:
                message += f" that serves {recommendation['food'].values[0]} food"
            message += ".\n"
            if information.inferences:
                message += information.inferences.message
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
            message += f" that serves {information.food} food"
        if information.inferences:
            message += f" that {information.inferences.consequent_sent}."
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
                value = recommendation[column]
                if isinstance(value, float):
                    if math.isnan(value):
                        message += f"has no {col_to_str[column]}"
                else:
                    message += f"has {col_to_str[column]}: {value}"
                if i + 1 < len(columns):
                    message += " and "
                i += 1
            message += ".\n"
        else:
            message = "I did not understand your request, please try again.\n"

        sentence = input(message)
        information = match_request(sentence, information)
        return sentence, information, recommendations


class RequestAdditionalInformation(StateInterface):
    def activate(self, information, recommendations):
        # User input
        INFERENCE_MAP = {
            "touristic": Inferences(
                "touristic",
                Inference(
                    "touristic",
                    True,
                    "is",
                    "it serves cheap, good food",
                    pricerange="cheap",
                    food_quality="good",
                ),
                Inference(
                    "touristic",
                    False,
                    "is not",
                    "it serves Romanian food",
                    food_type="romanian",
                ),
            ),
            "assigned seats": Inferences(
                "assigned seats",
                Inference(
                    "assigned seats",
                    True,
                    "has",
                    "the waiter decides where you sit",
                    crowdedness="busy",
                ),
            ),
            "children": Inferences(
                "children",
                None,
                Inference(
                    "children",
                    False,
                    "is not recommended for",
                    "spending a long time is not advised when taking children",
                    length_of_stay="long",
                ),
            ),
            "romantic": Inferences(
                "romantic",
                Inference(
                    "romantic",
                    True,
                    "is",
                    "spending a long time in a restaurant is romantic",
                    length_of_stay="long",
                ),
                Inference(
                    "romantic",
                    False,
                    "is not",
                    "a busy restaurant is not romantic",
                    crowdedness="busy",
                ),
            ),
        }
        sentence = input("Do you have any additional requirements? \n")
        consequent, truth_value = match_consequent(sentence)
        if consequent is not None and consequent in INFERENCE_MAP:
            inferences = INFERENCE_MAP[consequent]
            recommendations = inferences.infer(recommendations, truth_value)
            information.inferences = inferences
        return sentence, information, recommendations


def query(data, expected):
    SKIP = {"all", "any"}
    col, value = expected
    if value not in SKIP:
        return data[data[col] == value]
    return data


def query_information(data, information):
    data = data.copy()
    if information.pricerange:
        data = query(data, ("pricerange", information.pricerange))
    if information.area:
        data = query(data, ("area", information.area))
    if information.food:
        data = query(data, ("food", information.food))
    return data


def infer(data, inferences):
    for inference in inferences:
        data = inference.infer_from_data(data)
    return data


def get_information(sentence):
    return Information(
        match_pricerange(sentence), match_area(sentence), match_food(sentence)
    )


bye = ByeState(8, None, end=True)

extra_info = RequestInformation(7, None)
recommend = RecommendPlaceState(6, {"bye": bye, "request": extra_info, "thankyou": bye})
additional_info = RequestAdditionalInformation(8, recommend)

not_found = NotFoundState(5, None)
type_food = AskTypeState(4, {"inform": additional_info})
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
    transition(welcome)
