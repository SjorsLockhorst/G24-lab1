import re
from Levenshtein import distance

"""Funtions that takes a user's turn (type string) as input after the relevant question.
Checks whether it contains any information that can be used to fill slots.
If so, assigns that slot. If no type is recognized, should ask the user again."""


def match_by_keywords(sentence, keywords):
    # This is not the best, since it matches any word,
    # word that is in our query could have another semantic meaning to the user
    sentence = sentence.lower().strip()
    keyword_regex = f"({'|'.join(keywords)})"
    result = re.search(keyword_regex, sentence)
    if result:
        return result.group(1)

    # TODO: Match don't care, any, whatever no preference, then return "ANY".


def match_request(sentence, information):
    information.reset_requests()
    if match_by_keywords(sentence, ["pricerange"]):
        information.pricerange_requested = True
    if match_by_keywords(sentence, ["food"]):
        information.food_requested = True
    if match_by_keywords(sentence, ["area"]):
        information.area_requested = True
    if match_by_keywords(sentence, ["address"]):
        information.address_requested = True
    if match_by_keywords(sentence, ["postcode"]):
        information.postcode_requested = True
    if match_by_keywords(sentence, ["phone"]):
        information.phone_requested = True
    return information


def match_pricerange(sentence):
    sentence = sentence.lower().strip()
    PATTERN = r"\b(\w+)\s(priced|pricing|price)\b"
    KNOWN_RANGES = {"cheap", "expensive", "moderate"}
    match = match_sentence(sentence, PATTERN, KNOWN_RANGES, group=1)
    if not match:
        return match_by_keywords(sentence, KNOWN_RANGES)
    return match


def match_area(sentence):
    sentence = sentence.lower().strip()
    KNOWN_AREAS = {"west", "north", "south", "centre", "east"}
    FIRST_PATT = r"\b(\w+)\spart\b"
    SECOND_PATT = r"(in the|somewhere)\s(\w+)"
    first_match = match_sentence(sentence, FIRST_PATT, KNOWN_AREAS, group=1)
    if not first_match:
        second_match = match_sentence(sentence, SECOND_PATT, KNOWN_AREAS, group=2)
        if not second_match:
            return match_by_keywords(sentence, KNOWN_AREAS)
        return second_match
    return first_match


def match_food(sentence):
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
    match = match_sentence(sentence, PATTERN, KNOWN_FOODS)
    if not match:
        return match_by_keywords(sentence, KNOWN_FOODS)
    return match


def is_close_to_any(word, known_words, minimum_dist=3):
    return [
        known_word
        for known_word in known_words
        if distance(word, known_word) <= minimum_dist
    ]


def match_sentence(sentence, pattern, known_words, group=0):
    match = re.search(pattern, sentence)
    if match:
        matched_word = match.group(group)
        if matched_word in known_words:
            return matched_word

        corrections = is_close_to_any(matched_word, known_words)
        for correction in corrections:
            response = input(
                f"Didn't recognize {matched_word}, did you mean {correction}? (yes/no) \n"
            )
            if response == "yes":
                return correction


def match_consequent(sentence):
    KEYWORDS = ["touristic", "assigned seats", "children", "romantic"]
    return match_by_keywords(sentence, KEYWORDS)
