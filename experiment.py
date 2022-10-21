import os
import sys
from dataclasses import dataclass
from typing import List, Callable, Optional
from datetime import datetime
import random
import time

from rich import print
from rich.table import Table
from rich.console import Console
import pandas as pd

from dialog_system import start
from extract import read_augmented_restaurant_dataset

console = Console()
RESULTS_DIR = "results/experiment"

QUERY_PATH = os.path.join(RESULTS_DIR, "queries.csv")
RESPID_PATH = os.path.join(RESULTS_DIR, "respondents.csv")

respid_data = pd.read_csv(RESPID_PATH)
queries_data = pd.read_csv(QUERY_PATH)


def df_to_table(
    pandas_dataframe: pd.DataFrame,
    rich_table: Table = Table(),
    show_index: bool = True,
    index_name: Optional[str] = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj.
    Args:
        pandas_dataframe (DataFrame): A Pandas DataFrame to be converted to a rich Table.
        rich_table (Table): A rich Table that should be populated by the DataFrame values.
        show_index (bool): Add a column with a row count to the table. Defaults to True.
        index_name (str, optional): The column name to give to the index column. Defaults to None, showing no value.
    Returns:
        Table: The rich Table instance passed, populated with the DataFrame values."""

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(index)] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    return rich_table


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
    user_prompt: str
    query_tasks: List[QueryTask]
    run_experiment: Callable
    is_dialog_system: bool

    def shuffle_tasks(self):
        random.shuffle(self.query_tasks)

    def run(self, respid, shuffle=True):
        print(self.user_prompt)
        setup()
        if shuffle:
            self.shuffle_tasks()
        for i, task in enumerate(self.query_tasks, 1):
            print(f"Task [{i}/{len(self.query_tasks)}]")

            (
                selected_id,
                time_in_seconds,
                n_levenshtein,
                is_correct,
            ) = self.run_experiment(task)
            result = [
                respid,
                task.id,
                self.is_dialog_system,
                time_in_seconds,
                selected_id,
                is_correct,
                n_levenshtein,
            ]
            queries_data.loc[len(queries_data)] = result


raw_data = read_augmented_restaurant_dataset()


# Randomize which tasks are performed first, raw data / dialog system

TOURISTIC_TEXT = (
    "A restaurant is touristic when it is cheap in price and serves good food."
)
ROMANTIC_TEXT = "A restaurant is romantic when you can stay a long time."

raw_query_tasks: List[QueryTask] = [
    QueryTask(
        1,
        "Find a cheap place in the south, any food type, any pricerange.",
        [10, 14, 37, 43, 54],
    ),
    QueryTask(
        2,
        "Find a European place in the centre, any pricerange.",
        [26, 32, 47, 96, 106],
    ),
    QueryTask(
        3,
        "Find a touristic place in the centre that serves Polynesian food, any pricerange.",
        [66],
        TOURISTIC_TEXT,
    ),
    QueryTask(
        4,
        "Find a romantic place in the west part of town, any food type, any pricerange.",
        [46],
        ROMANTIC_TEXT,
    ),
    QueryTask(
        5,
        "Find a cheap, Italian place in the east part of town. If there is none, try anywhere else in town.",
        [37, 102],
    ),
]

dialog_system_query_tasks: List[QueryTask] = [
    QueryTask(
        1,
        "Find an expensive place in the north, any food type.",
        [7, 8, 9, 39, 94],
    ),
    QueryTask(
        2,
        "Find a Chinese place in the south, any pricerange.",
        [11, 14, 43, 49, 54],
    ),
    QueryTask(
        3,
        "Find a touristic place in the south that serves Chinese food, any pricerange.",
        [43],
    ),
    QueryTask(
        4,
        "Find a romantic place in the north part of town, any food type any pricerange.",
        [1],
    ),
    QueryTask(
        5,
        "Find a place that has a moderate pricerange, that serves modern European in the west part of town. If there is none, try anywhere else in town.",
        [1, 51],
    ),
]


def setup(clear=True):
    answer = ""
    while answer != "yes":
        answer = input("Are you ready? (type yes and press Enter to continue): ")
    if clear:
        console.clear()


def start_dialog_system(query_task: QueryTask):
    print(query_task.user_prompt)
    setup(clear=False)
    if query_task.extra_info:
        print(query_task.extra_info)
    print()
    selected_id, time_in_seconds, n_levenshtein = start()
    is_correct = selected_id in query_task.correct_rows
    console.clear()
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

    # What restaurant did the user pick?
    # TODO: Let user type in restaurant id
    # How long did it take?
    print(query_task.user_prompt)
    if query_task.extra_info:
        print(query_task.extra_info)
    setup()
    print()
    print(query_task.user_prompt)
    print(table)
    print(query_task.user_prompt)
    if query_task.extra_info:
        print(query_task.extra_info)
    print()
    selected_id = print(
        "Fill in restaurant id (number in first column) and press Enter to confirm: ",
        end="",
    )
    valid = False
    start = time.time()
    while not valid:
        selected_id = input()
        try:
            selected_id_int = int(selected_id)
        except ValueError:
            print("Please enter a valid number: ", end="")
        else:
            if selected_id_int < 0 or selected_id_int >= len(df):
                print(f"Please enter a number between 0 and {len(df)}: ", end="")
            else:
                valid = True
    end = time.time()
    time_in_seconds = end - start
    console.clear()
    return (
        selected_id_int,
        time_in_seconds,
        None,
        selected_id_int in query_task.correct_rows,
    )


dialog_experiment = Experiment(
    "Dialog system experiment",
    "You will now search for restaurants using the dialog system. Please read the instructions carefully before starting.",
    dialog_system_query_tasks,
    start_dialog_system,
    True,
)
raw_data_experiment = Experiment(
    "Raw data search experiment",
    "You will now search for restaurants by doing a manual search. Please read the instructions carefully before starting.",
    raw_query_tasks,
    start_raw_data_search,
    False,
)

EXPERIMENTS = [dialog_experiment, raw_data_experiment]


def validate_int(val, lower, upper):
    try:
        integer = int(val)
    except ValueError:
        return False
    else:
        if integer < lower or integer > upper:
            return False
    return True


def run_experiment(experiments=EXPERIMENTS):
    respid = -1
    which_exp_first = input(
        "Which experiment to run first: [1] dialog system [2] manual search: "
    )
    while not validate_int(which_exp_first, 1, 2):
        which_exp_first = input("Please pick 1 or 2! ")
    which_exp_first_int = int(which_exp_first)
    if which_exp_first_int == 2:
        experiments.reverse()

    respid_str = input("Please enter respondent id: ")
    while not validate_int(respid_str, 0, sys.maxsize):
        respid_str = input("Please enter a valid non negative integer")

    respid = int(respid_str)
    if respid in respid_data["resp_id"].values:
        raise Exception(
            f"Respid {respid} already exists in data, please re run with a unique respid."
        )
    console.clear()
    print("Welcome to our experiment!\n")

    birthyear_str = input("What year where you born in: ")
    while not validate_int(birthyear_str, 1900, 2022):
        birthyear_str = input("Please provide a year between 1900, and 2022. ")
    console.clear()
    birthyear = int(birthyear_str)
    age = datetime.now().year - birthyear
    respid_data.loc[len(respid_data)] = (respid, age)
    for exp in experiments:
        exp.run(respid)
    queries_data.to_csv(QUERY_PATH, index=False)
    respid_data.to_csv(RESPID_PATH, index=False)


if __name__ == "__main__":
    run_experiment()
