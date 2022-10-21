"""Templates to extract information from user inputs."""

import re
from Levenshtein import distance


def match_by_keywords(sentence, keywords, information, use_levenshtein=False):
    """Match keywords in a sentence."""
    # TODO: Match don't care, any, whatever no preference, then return "ANY".
    sentence = sentence.lower().strip()
    keyword_regex = rf"\b({'|'.join(keywords)})\b"
    result = re.search(keyword_regex, sentence)
    words = sentence.split()
    if len(words) == 1:
        if words[0] in {"all", "any"}:
            return "any"
    if result:
        return result.group(1)
    if use_levenshtein:
        for word in words:
            corrections = is_close_to_any(word, keywords)
            for correction in corrections:
                response = input(
                    f"Didn't recognize {word}, did you mean {correction}? (yes/no) \n"
                )
                if response == "yes":
                    information.n_levenshtein += 1
                    return correction


def match_request(sentence, information):
    """Match which request a user has typed in a sentence."""
    information.reset_requests()
    if match_by_keywords(sentence, ["pricerange"], information):
        information.pricerange_requested = True
    if match_by_keywords(sentence, ["food"], information):
        information.food_requested = True
    if match_by_keywords(sentence, ["area"], information):
        information.area_requested = True
    if match_by_keywords(sentence, ["address"], information):
        information.address_requested = True
    if match_by_keywords(sentence, ["postcode"], information):
        information.postcode_requested = True
    if match_by_keywords(sentence, ["phone"], information):
        information.phone_requested = True
    return information


def match_pricerange(sentence, information, use_levenshtein_keywords=True):
    """Matches the template for pricerange against a user input."""
    sentence = sentence.lower().strip()
    PATTERN = r"\b(\w+)\s(priced|pricing|price|pricerange)\b"
    KNOWN_RANGES = {"cheap", "expensive", "moderate"}
    match = match_template(sentence, PATTERN, KNOWN_RANGES, information, group=1)
    if not match:
        return match_by_keywords(
            sentence,
            KNOWN_RANGES,
            information,
            use_levenshtein=use_levenshtein_keywords,
        )
    return match


def match_area(sentence, information, use_levenshtein_keywords=True):
    """Matches the template for area against a user input."""
    sentence = sentence.lower().strip()
    KNOWN_AREAS = {"west", "north", "south", "centre", "east"}
    FIRST_PATT = r"\b(\w+)\s(part|area)\b"
    SECOND_PATT = r"(in the|somewhere)\s(\w+)"
    first_match = match_template(
        sentence, FIRST_PATT, KNOWN_AREAS, information, group=1
    )
    if not first_match:
        second_match = match_template(
            sentence, SECOND_PATT, KNOWN_AREAS, information, group=2
        )
        if not second_match:
            return match_by_keywords(
                sentence,
                KNOWN_AREAS,
                information,
                use_levenshtein=use_levenshtein_keywords,
            )
        return second_match
    return first_match


def match_food(sentence, information, use_levenshtein_keywords=True):
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
    match = match_template(sentence, PATTERN, KNOWN_FOODS, information, group=1)
    if not match:
        return match_by_keywords(
            sentence, KNOWN_FOODS, information, use_levenshtein=use_levenshtein_keywords
        )
    return match


def is_close_to_any(word, known_words, minimum_dist=3):
    """From a list of words, find those words who are close to some known words."""
    try:
        results = [
            known_word
            for known_word in known_words
            if distance(word, known_word) <= minimum_dist
        ]
    # AAAHHHHHH
    except Exception:
        return []
    return results


def match_template(sentence, pattern, known_words, information, group=0):
    """Match a pattern and known words against a user input."""
    match = re.search(pattern, sentence)
    if match:
        matched_word = match.group(group)
        if matched_word in known_words:
            return matched_word
        if matched_word in {"all", "any"}:
            return matched_word

        corrections = is_close_to_any(matched_word, known_words)
        for correction in corrections:
            response = input(
                f"Didn't recognize {matched_word}, did you mean {correction}? (yes/no) \n"
            )
            if response == "yes":
                information.n_levenshtein += 1
                return correction


def match_consequent(sentence, information):
    """Match which consequent a user inputs."""
    KEYWORDS = ["touristic", "assigned seats", "children", "romantic"]
    NEGATIVE_KEYWORDS = ["not", "no"]
    match = match_by_keywords(sentence, KEYWORDS, information, True)
    return match, match_by_keywords(sentence, NEGATIVE_KEYWORDS, information) is None
