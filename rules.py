"""Module that implements different keyword matching rules"""
import re


def is_goodbye(sentence):
    return re.match(r"^.*(good? bye)|(\wbye\w).*$", sentence)


def is_thank_you(sentence):
    return re.match(r"^.*thank.*$", sentence)


def is_request(sentence):
    return re.match(r"^.*(what|address|phone|number|post? code|zip? code).*$", sentence)


def is_reqalts(sentence):
    return re.match(r"^.*(what|how) about.*$", sentence)


def is_affirm(sentence):
    return re.match(r"^.*(yes|correct).*$", sentence)


def is_inform(sentence):
    return re.match(r"^.*(cheap|price|expensive).*$", sentence)


def is_negate(sentence):
    return re.match(r"^.*(not|no).*$", sentence)


def is_null(sentence):
    return re.match(r"^.*(unintellgible|noisy|cough|tv_noise).*$", sentence)


DIALOG_ACT_RULE_MAPPING = [
    (is_thank_you, "thankyou"),
    (is_request, "request"),
    (is_reqalts, "reqalts"),
    (is_affirm, "affirm"),
    (is_inform, "inform"),
    (is_negate, "negate"),
    (is_goodbye, "bye"),
    (is_null, "null"),
]
