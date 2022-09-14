"""Module that implements different keyword matching rules"""
import re


def is_goodbye(sentence):
    """Matches goodbye rule."""
    return re.match(r"^.*good? bye.*$", sentence)


def is_thank_you(sentence):
    """Mathces thank you rule."""
    return re.match(r"^.*thank? you.*$", sentence)


DIALOG_ACT_RULE_MAPPING = [(is_goodbye, "bye"), (is_thank_you, "thankyou")]
