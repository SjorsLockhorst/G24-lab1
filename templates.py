import re
from Levenshtein import distance

"""Funtions that takes a user's turn (type string) as input after the relevant question.
Checks whether it contains any information that can be used to fill slots.
If so, assigns that slot. If no type is recognized, should ask the user again."""


def match_sentence(sentence, keywords):
    matched = _match_sentence(sentence, keywords)
    if not matched:
        words = sentence.split(" ")
        for word in words:
            for keyword in keywords:
                if distance(word, keyword) < 3:
                    response = input(
                        f"Didn't recognize {word}, did you mean {keyword}? (yes/no) \n"
                    )
                    if response == "yes":
                        return keyword
    return matched


def _match_sentence(sentence, keywords):
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
    if match_sentence(sentence, ["pricerange"]):
        information.pricerange_requested = True
    if match_sentence(sentence, ["food"]):
        information.food_requested = True
    if match_sentence(sentence, ["area"]):
        information.area_requested = True
    if match_sentence(sentence, ["address"]):
        information.address_requested = True
    if match_sentence(sentence, ["postcode"]):
        information.postcode_requested = True
    if match_sentence(sentence, ["phone"]):
        information.phone_requested = True
    return information


def match_pricerange(sentence):
    return match_sentence(sentence, ["cheap", "expensive", "moderate"])


def match_area(sentence):
    return match_sentence(sentence, ["west", "north", "south", "centre", "east"])


def match_food(sentence):
    return match_sentence(
        sentence,
        [
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
        ],
    )


# def find_pricerange(user_input):
#     user_input = user_input.lower().strip()

#     slot_filled = False  # First check whether a user explicitly mentions a label
#     for word in user_input:
#         if word in pricerange_labels:
#             # fill pricerange slot with 'word'
#             slot_filled = True

#     if not slot_filled:
#         template = re.compile()
#         mo = re.search(r"\b\w+\s(priced|pricing|price ?(range))\b", user_input)
#         if mo:
#             # fill pricerange slot with "mo.group().split()[0]"
#             SlotFilled = True

#     if not SlotFilled:
#         # ask user again.
#         pass

#     return


# def find_area(user_input):
#     user_input = user_input.strip.lower()

#     SlotFilled = False  # First check whether a user explicitly mentions a label
#     for word in user_input:
#         if word in area_labels:
#             # fill area slot with 'word'
#             SlotFilled = True

#     if not SlotFilled:
#         AreaTemplate = re.compile(r"\b\w+\s(part)\b")
#         mo = regex.search(user_input)
#         if mo:
#             # fill pricerange slot with "mo.group().split()[0]"
#             SlotFilled = True

#     if not SlotFilled:
#         AreaTemplate2 = re.compile(r"\b(in the|somewhere)\s\w+\b")
#         mo = regex.search(user_input)
#         if mo:
#             # fill pricerange slot with "mo.group().split()[-1]"
#             SlotFilled = True

#     if not SlotFilled:
#         # ask user again.
#         pass

#     return


# def find_food(user_input):
#     user_input = user_input.strip.lower()

#     SlotFilled = False  # First check whether a user explicitly mentions a label
#     for word in user_input:
#         if word in food_labels:
#             # fill food slot with 'word'
#             SlotFilled = True

#     if not SlotFilled:
#         FoodTemplate = re.compile(r"\b\w+\s(food|cuisine|kitchen|restaurant|place)\b")
#         mo = regex.search(user_input)
#         if mo:
#             # fill pricerange slot with "mo.group().split()[0]"
#             SlotFilled = True

#     if not SlotFilled:
#         pass
#         # ask user again.

#     return
