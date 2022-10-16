from dataclasses import dataclass
from typing import List, Callable, Optional
from datetime import datetime
import random
import time

from rich import print
from rich.console import Console
from rich_tools import df_to_table

from dialog_system import start
from extract import read_augmented_restaurant_dataset

console = Console()


@dataclass
class QueryTask:
    id: int
    user_prompt: str
    correct_rows: List[int]
    extra_info: Optional[str] = None


# TODO: Add a prompt to introduce the overall experiment
@dataclass
class Experiment:
    name: str
    query_tasks: List[QueryTask]
    run_experiment: Callable

    def shuffle_tasks(self):
        random.shuffle(self.query_tasks)

    def run(self, shuffle=True):
        data = []
        if shuffle:
            self.shuffle_tasks()
        for task in self.query_tasks:
            result = self.run_experiment(task)
            data.append(result)
            print(result)
        return data


raw_data = read_augmented_restaurant_dataset()


# Randomize which tasks are performed first, raw data / dialog system

TOURISTIC_TEXT = (
    "A restaurant is touristic when it is cheap in price and serves good food."
)
ROMANTIC_TEXT = "A restaurant is romantic when you can stay a long time."

raw_query_tasks: List[QueryTask] = [
    QueryTask(
        1, "Find a cheap place in the south, any food type.", [10, 14, 37, 43, 54]
    ),
    QueryTask(2, "Find a European place in the centre.", [26, 32, 47, 96, 106]),
    QueryTask(
        3,
        "Find a touristic place in the centre that serves Polynesian food.",
        [66],
        TOURISTIC_TEXT,
    ),
    QueryTask(
        4, "Find a romantic place in the west part of town.", [46], ROMANTIC_TEXT
    ),
    QueryTask(
        5,
        "Find a cheap, Italian place in the east part of town. If there is none, try anywhere else in town.",
        [37, 102],
    ),
]

dialog_system_query_tasks: List[QueryTask] = [
    QueryTask(
        1, "Find an expensive place in the north, any food type.", [7, 8, 9, 39, 94]
    ),
    QueryTask(2, "Find a Chinese place in the south.", [11, 14, 43, 49, 54]),
    QueryTask(3, "Find a touristic place in the south that serves Chinese food.", [43]),
    QueryTask(4, "Find a romantic place in the north part of town.", [1]),
    QueryTask(
        5,
        "Find a moderately priced, modern European place in the west part of town. If there is none, try anywhere else in town.",
        [1, 51],
    ),
]


def setup():
    answer = ""
    while answer != "yes":
        answer = input("Are you ready? (type yes to continue)")


def start_dialog_system(query_task: QueryTask):
    # What restaurant did the user pick?
    # How long did it take?
    # How many Levenshtein / backspaces etc. (extra metrics)
    # TODO: implement in dialog system, that it can return the user pick
    # TODO: inform user they have to type bye, to select their restaurant
    setup()
    selected_id, time_in_seconds, n_levenshtein = start()
    is_correct = selected_id in query_task.correct_rows
    return selected_id, time_in_seconds, n_levenshtein, is_correct


COLUMN_MAP = {
    "restaurantname": "Name",
    "area": "Area",
    "pricerange": "Price range",
    "food": "Food type",
    "food quality": "Food quality",
    "crowdedness": "Crowdedness",
    "length of stay": "Length of stay",
}
df = raw_data[COLUMN_MAP.keys()].rename(columns=COLUMN_MAP)
df.index.name = "id"

table = df_to_table(df)


def start_raw_data_search(query_task: QueryTask, table=table):

    console.clear()
    # What restaurant did the user pick?
    # TODO: Let user type in restaurant id
    # How long did it take?
    print(query_task.user_prompt)
    if query_task.extra_info:
        print(query_task.extra_info)
    setup()
    print(table)
    print(query_task.user_prompt)
    if query_task.extra_info:
        print(query_task.extra_info)
    selected_id = print("Restaurant id? ")
    valid = False
    start = time.time()
    while not valid:
        selected_id = input()
        try:
            selected_id_int = int(selected_id)
        except ValueError:
            print("Please enter a valid number.")
        else:
            if selected_id_int < 0 or selected_id_int >= len(df):
                print(f"Please enter a number between 0 and {len(df)}")
            else:
                valid = True
    end = time.time()
    time_in_seconds = end - start
    return (
        selected_id_int,
        time_in_seconds,
        None,
        selected_id_int in query_task.correct_rows,
    )


dialog_experiment = Experiment(
    "Dialog system experiment", dialog_system_query_tasks, start_dialog_system
)
raw_data_experiment = Experiment(
    "Raw data search experiment", raw_query_tasks, start_raw_data_search
)

EXPERIMENTS = [dialog_experiment, raw_data_experiment]


def run_experiment(experiments=EXPERIMENTS):
    # Welcome user to experiment
    # Explain anything that might be needed to perform experiment
    console.clear()
    print("Welcome to our experiment!\n")

    print("For this experiment, you are hungry, and you want to find a place to eat.")
    print(
        "Using different techniques, you will have to find some restaurants that might be suitable according to your needs."
    )
    print(
        "You will receive information about the type of restaurant you would like to eat at."
    )
    print("It is then up to you to select a restaurant that matches these preferences.")
    print(
        "Since you are hungry, you want to try to find a restaurant as quickly as possible, without accidentally picking the wrong one.\n"
    )
    print(
        "Let's start off with getting to know you a tiny bit, in what year where you born?\n"
    )
    valid = False
    while not valid:
        try:
            birthyear_int = int(input())
        except ValueError:
            print("Please enter a valid number.")
        else:
            if birthyear_int < 1900 or birthyear_int > 2022:
                print("Please enter a birthyear between 1900 and 2022")
            else:
                valid = True
    setup()

    age = datetime.now().year - birthyear_int
    random.shuffle(experiments)
    for exp in experiments:
        exp.run()


if __name__ == "__main__":
    run_experiment()
