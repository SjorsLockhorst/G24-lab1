"""Templates to extract information from user inputs."""

import re
from Levenshtein import distance

UNIVERSALS = {"all", "any"}


def match_by_keywords(sentence, keywords, use_levenshtein=False):
    """Match keywords in a sentence."""
    sentence = sentence.lower().strip()
    keywords = set(keywords) | UNIVERSALS
    keyword_regex = rf"\b({'|'.join(keywords)})\b"
    result = re.search(keyword_regex, sentence)
    if result:
        return result.group(1)
    if use_levenshtein:
        for word in sentence.split():
            corrections = is_close_to_any(word, keywords)
            for correction in corrections:
                response = input(
                    f"Didn't recognize {word}, did you mean {correction}? (yes/no) \n"
                )
                if response == "yes":
                    return correction


def match_request(sentence, information):
    """Match which request a user has typed in a sentence."""
    information.reset_requests()
    if match_by_keywords(sentence, {"pricerange"}):
        information.pricerange_requested = True
    if match_by_keywords(sentence, {"food"}):
        information.food_requested = True
    if match_by_keywords(sentence, {"area"}):
        information.area_requested = True
    if match_by_keywords(sentence, {"address"}):
        information.address_requested = True
    if match_by_keywords(sentence, {"postcode"}):
        information.postcode_requested = True
    if match_by_keywords(sentence, {"phone"}):
        information.phone_requested = True
    return information


def match_pricerange(sentence, use_levenshtein_keywords=True):
    """Matches the template for pricerange against a user input."""
    sentence = sentence.lower().strip()
    PATTERN = r"\b(\w+)\s(priced|pricing|price|pricerange)\b"
    KNOWN_RANGES = {"cheap", "expensive", "moderate"}
    match = match_template(sentence, PATTERN, KNOWN_RANGES, group=1)
    if not match:
        return match_by_keywords(
            sentence, KNOWN_RANGES, use_levenshtein=use_levenshtein_keywords
        )
    return match


def match_area(sentence, use_levenshtein_keywords=True):
    """Matches the template for area against a user input."""
    sentence = sentence.lower().strip()
    KNOWN_AREAS = {"west", "north", "south", "centre", "east"}
    FIRST_PATT = r"\b(\w+)\spart\b"
    SECOND_PATT = r"(in the|somewhere)\s(\w+)"
    first_match = match_template(sentence, FIRST_PATT, KNOWN_AREAS, group=1)
    if not first_match:
        second_match = match_template(sentence, SECOND_PATT, KNOWN_AREAS, group=2)
        if not second_match:
            return match_by_keywords(
                sentence, KNOWN_AREAS, use_levenshtein=use_levenshtein_keywords
            )
        return second_match
    return first_match


def match_food(sentence, use_levenshtein_keywords=True):
    """Matches the template for food against a user input."""
    sentence = sentence.lower().strip()
    KNOWN_FOODS = {
        "british",
        "modern european",
        "italian",
        "romanian",
        "seafood",
        "chinese",
        "steakhouse",
        "asian oriental",
        "french",
        "portuguese",
        "indian",
        "spanish",
        "european",
        "vietnamese",
        "korean",
        "thai",
        "moroccan",
        "swiss",
        "fusion",
        "gastropub",
        "tuscan",
        "international",
        "traditional",
        "mediterranean",
        "polynesian",
        "african",
        "turkish",
        "bistro",
        "north american",
        "australasian",
        "persian",
        "jamaican",
        "lebanese",
        "cuban",
        "japanese",
        "catalan",
    }

    PATTERN = r"\b(\w+)\sfood|cuisine|kitchen|restaurant|place\b"
    match = match_template(sentence, PATTERN, KNOWN_FOODS, group=1)
    if not match:
        return match_by_keywords(
            sentence, KNOWN_FOODS, use_levenshtein=use_levenshtein_keywords
        )
    return match


def is_close_to_any(word, known_words, minimum_dist=3):
    """From a list of words, find those words who are close to some known words."""
    return [
        known_word
        for known_word in known_words
        if distance(word, known_word) <= minimum_dist
    ]


def match_template(sentence, pattern, known_words, group=0):
    """Match a pattern and known words against a user input."""
    match = re.search(pattern, sentence)
    if match:
        matched_word = match.group(group)
        if matched_word in known_words:
            return matched_word
        if matched_word in UNIVERSALS:
            return matched_word

        corrections = is_close_to_any(matched_word, known_words)
        for correction in corrections:
            response = input(
                f"Didn't recognize {matched_word}, did you mean {correction}? (yes/no) \n"
            )
            if response == "yes":
                return correction


def match_consequent(sentence):
    """Match which consequent a user inputs."""
    KEYWORDS = {"touristic", "assigned seats", "children", "romantic"}
    NEGATIVE_KEYWORDS = {"not", "no"}
    match = match_by_keywords(sentence, KEYWORDS)
    return match, match_by_keywords(sentence, NEGATIVE_KEYWORDS) is None
