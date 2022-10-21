"""
A dialog system, that prompts user for restaurant specifications, and ultimates
attempts to suggest a restaurant that matches these specifications.
"""

import abc
import math
from typing import Optional
import time

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

NOT_FOUND = "NOT_FOUND"


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
        """Representation for this class, for debugging"""
        return f"{self.__class__.__name__}(number={self.number}, end={self.end})"


@dataclass
class Inference:
    """An inference that we want to make on our restaurant data."""

    consequent: str
    truth_value: bool

    # Variables to format strings to user correctly
    verb: str
    because: str

    # Possible antecedent values
    pricerange: Optional[str] = None
    food_quality: Optional[str] = None
    length_of_stay: Optional[str] = None
    food_type: Optional[str] = None
    crowdedness: Optional[str] = None

    def infer_from_data(self, recommendations):
        """Infer from the data which recommendations meet our antecedent."""
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
        """Get the sentence that involves the consequent."""
        return f"{self.verb} {self.consequent}"

    def __str__(self):
        """
        Format this inference based on the consequent sent and the because properties
        """
        return f"The restaurant {self.consequent_sent} because {self.because}.\n"


class Inferences:
    """
    A collection of at least one and at most two inferences.

    One can define a inference when true, and inference when false.

    Raises a ValueError when both inferences are None.
    """

    def __init__(self, consequent, true_inference=None, false_inference=None):
        self.consequent = consequent
        self.true_inference = true_inference
        self.false_inference = false_inference
        self.truth_value = None
        if not true_inference and not false_inference:
            raise ValueError("Must have at least one inference")

    def infer(self, recommendations, truth_value):
        """
        Infer from the data the correct inference, based on the truth value
        """
        # Copy the current recommendations
        new_rec = recommendations.copy()

        # Set the truth value of this inference
        self.truth_value = truth_value

        # If the truth_value == True
        if truth_value:

            # If we have both an inference for True and False values
            if self.true_inference is not None and self.false_inference is not None:

                # Infer which restaurants in the data have value True
                new_rec = self.true_inference.infer_from_data(new_rec)

                # Infer which restaurants in the data have value False
                to_remove = self.false_inference.infer_from_data(new_rec)

                # Drop the resturants where the inference is False
                # because we want only restaurants that are True
                new_rec = new_rec.drop(index=to_remove.index)

            # If we have only the true inference
            elif self.true_inference is not None and self.false_inference is None:

                # See for which
                new_rec = self.true_inference.infer_from_data(new_rec)

            # If we have only a False inference, but the truth value is True,
            # we can't resolve the request (we don't know what it means for our
            # inference to be True) hence, we return no results
            elif self.true_inference is None and self.false_inference is not None:
                new_rec = pd.DataFrame()

        # Same comments as before apply, but now inversely, where True is now False
        else:
            if self.false_inference is not None and self.true_inference is not None:
                new_rec = self.false_inference.infer_from_data(new_rec)
                to_remove = self.true_inference.infer_from_data(new_rec)
                new_rec = new_rec.drop(index=to_remove.index)
            elif self.false_inference is not None and self.false_inference is None:
                new_rec = self.false_inference.infer_from_data(new_rec)
            elif self.false_inference is None and self.true_inference is not None:
                new_rec = pd.DataFrame()

        # Finally, return recommendations for which the resturants match the inference
        # with the correct truth value.
        return new_rec

    @property
    def chosen_inference(self):
        """The inference that was chosen, based on the truth value that was set."""
        if self.truth_value is None:
            return None
        if self.truth_value:
            return self.true_inference
        else:
            return self.false_inference

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


@dataclass
class Information:
    """Models the information that a user can give us via inputted sentences"""

    pricerange: Optional[str]
    area: Optional[str]
    food: Optional[str]

    postcode_requested: bool = False
    address_requested: bool = False
    phone_requested: bool = False
    pricerange_requested: bool = False
    area_requested: bool = False
    food_requested: bool = False
    inferences: Optional[Inferences] = None

    n_levenshtein: int = 0

    def reset_requests(self):
        """Reset all requests."""
        self.postcode_requested = False
        self.address_requested = False
        self.phone_requested = False
        self.pricerange_requested = False
        self.area_requested = False
        self.food_requested = False

    def update(self, other):
        """Update information with another information object."""
        if other.pricerange:
            self.pricerange = other.pricerange
        if other.area:
            self.area = other.area
        if other.food:
            self.food = other.food

    def get_requested_columns(self):
        """Get columns from restaurant data that matches current information."""
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


class WelcomeState(StateInterface):
    """The state that welcomes the user, and asks for the first user input."""

    def activate(self, information, recommendations):
        sentence = input(
            "Hello , welcome to the Cambridge restaurant system? You can ask for "
            "restaurants by area , price range or food type . How may I help you?\n"
        )
        # Extract information from sentence
        new_information = get_information(sentence)

        new_recommendations = query_information(data, new_information)

        return sentence.lower(), new_information, new_recommendations


class ByeState(StateInterface):
    """The state that says goodbye to the user."""

    def activate(self, information, recommendations):
        print("Good bye\n")
        return "", information, recommendations


class AskPriceRangeState(StateInterface):
    """The state that asks the user for a price range preference"""

    def activate(self, information, recommendations):
        sentence = ""
        # Only ask for pricerange, if we don't have one unique pricerange yet
        # in our recommendations
        if len(recommendations["pricerange"].unique()) > 1:
            # Loop while we don't know the users preference for pricerange
            while not information.pricerange:
                sentence = input(
                    "Would you like something in the cheap, moderate or expensive price range?\n"
                )
                # Store the new value that was matched from the user input in information
                information.pricerange = match_pricerange(sentence, information)

        # Query recommendations based on new information
        new_recommendations = query_information(data, information)
        information.inferences = None

        return sentence.lower(), information, new_recommendations


class AskTypeState(StateInterface):
    """State that asks the user for the type of food they would like."""

    def activate(self, information, recommendations):
        sentence = ""
        # Only ask for food type, if we don't have a unique food yet in recommendation
        if len(recommendations["food"].unique()) > 1:
            while not information.food:
                sentence = input("What kind of food would you like?\n")
                information.food = match_food(sentence, information)
        new_recommendations = query_information(data, information)
        return sentence.lower(), information, new_recommendations


class AskAreaState(StateInterface):
    """State that asks the user for the area of town they would prefer."""

    def activate(self, information, recommendations):
        sentence = ""
        if len(recommendations["area"].unique()) > 1:
            while not information.area:
                sentence = input("What kind of area would you like?\n")
                information.area = match_area(sentence, information)
        new_recommendations = query_information(data, information)
        return sentence.lower(), information, new_recommendations


class RecommendPlaceState(StateInterface):
    """State that picks a restaurant recommendation for the user"""

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
            print(new_recommendations)
            return sentence.lower(), new_information, new_recommendations
        return NOT_FOUND, information, new_recommendations


class NotFoundState(StateInterface):
    """
    State that let's the user know there are no recommendations that meet
    their requirements.
    """

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
        message += "\nPlease adjust your requirements to find a restaurant.\n"
        sentence = input(message)
        information.update(get_information(sentence))
        return sentence.lower(), information, data


class RequestInformation(StateInterface):
    """State that handles when user asks for more information."""

    def activate(self, information, recommendations):
        recommendation = recommendations.loc[len(data)]
        columns = information.get_requested_columns()

        # Mapping from column in data to natural language
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
            # Loop over each column that was requested by the user
            while i < len(columns):
                column = columns[i]
                value = recommendation[column]

                # If there is no value for the column in the data, let the user know
                if isinstance(value, float):
                    if math.isnan(value):
                        message += f"has no {col_to_str[column]}"
                else:
                    # Otherwise, print the value
                    message += f"has {col_to_str[column]}: {value}"
                # If we're not in the last iteration, print and
                if i + 1 < len(columns):
                    message += " and "
                i += 1
            message += ".\n"
        else:
            message = "I did not understand your request, please try again.\n"

        sentence = input(message)
        information = match_request(sentence, information)
        return sentence.lower(), information, recommendations


class AskAdditionalRequirements(StateInterface):
    """
    State that handles additional requirements from the user,
    based on inference rules.
    """

    def activate(self, information, recommendations):
        # Map each known consequent string that a user might type
        # to an associated set of inferences.
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
        if len(recommendations) == 0:
            return NOT_FOUND, information, recommendations
        sentence = input("Do you have any additional requirements? \n")
        consequent, truth_value = match_consequent(sentence, information)
        if consequent is not None and consequent in INFERENCE_MAP:
            inferences = INFERENCE_MAP[consequent]
            recommendations = inferences.infer(recommendations, truth_value)
            information.inferences = inferences
        return sentence.lower(), information, recommendations


def query(data, expected):
    """Query the data, skip if value for column is in SKIP"""
    SKIP = {"all", "any"}
    col, value = expected
    if value not in SKIP:
        return data[data[col] == value]
    return data


def query_information(data, information):
    """Query the data based on some given information."""
    data = data.copy()
    if information.pricerange:
        data = query(data, ("pricerange", information.pricerange))
    if information.area:
        data = query(data, ("area", information.area))
    if information.food:
        data = query(data, ("food", information.food))
    return data


def get_information(sentence):
    """Update information based on a user input."""
    information = Information(None, None, None)
    information.pricerange = match_pricerange(sentence, information, False)
    information.area = match_area(sentence, information, False)
    information.food = match_food(sentence, information, False)
    return information


# Collection of states, connected together as shown in the diagram
bye = ByeState(9, None, end=True)
extra_info = RequestInformation(8, None)
recommend = RecommendPlaceState(7, {"bye": bye, "request": extra_info, "thankyou": bye})
additional_info = AskAdditionalRequirements(6, recommend)

not_found = NotFoundState(5, None)
type_food = AskTypeState(4, {"inform": additional_info})
ask_area = AskAreaState(3, {"inform": type_food})
price_range = AskPriceRangeState(2, {"inform": ask_area})

not_found.next_state = {"inform": price_range}
recommend.next_state["NOT_FOUND"] = not_found
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
    """
    Transition function for dialog management system.

    Calls itself recursively untill it reaches some state for which end=True.
    Activates each state, and classifies the returned sentence using some
    classification model. Passes on all information and recommendations that were
    generated by each state to the next state. Knows what state to transition to,
    by transitioning to the correct state.next_state, based on dialog act.
    Some states have only one possible next state, in this case the transition function
    will always pick this one as the next.
    """
    sentence, updated_information, new_recommendations = state.activate(
        information, recommendations
    )
    dialog_act = model.predict([sentence.lower()])[0]

    if not state.end:

        if sentence != NOT_FOUND:
            # Only if we have a transition we can make, use this next state
            if isinstance(state.next_state, dict):

                # Get dialog act to find out to which state to transition

                if dialog_act in state.next_state:
                    next_state = state.next_state[dialog_act]

                # If there's no next state corresponding to dialog act, repeat this
                # state.
                else:
                    next_state = state
            else:
                next_state = state.next_state
        else:
            next_state = not_found

        # Verbosity code, toggeling how much information to print to the user.
        # Used to debug the code, and to see the path taken through the diagram.
        if verbose:
            print(f"Dialog act: {dialog_act}")
            print(f"Previous state: {state}")
            print(f"Next state: {next_state}")
            print(f"Current information: {updated_information}")
            print(f"Recommended based on information: {new_recommendations}")

        # Recursively call this function with updated information
        return transition(
            next_state,
            information=updated_information,
            recommendations=new_recommendations,
            model=model,
            verbose=verbose,
        )
    else:
        if len(data) in recommendations.index:
            selected = recommendations.loc[len(data)]
            return selected.original_index, information


def start():
    start = time.time()
    selected_id, information = transition(welcome)
    end = time.time()
    time_in_seconds = end - start
    return selected_id, time_in_seconds, information.n_levenshtein


if __name__ == "__main__":
    # Activate first state
    transition(welcome, verbose=True)
