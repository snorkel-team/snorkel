from snorkel.labeling.preprocess import preprocessor
from snorkel.types import DataPoint


def get_person_text(cand):
    """
    Returns the text for the two person mentions in candidate cand
    """
    person_names = []
    for index in [1, 2]:
        field_name = "person{}_word_idx".format(index)
        start = cand[field_name][0]
        end = cand[field_name][1] + 1
        person_names.append(" ".join(cand["tokens"][start:end]))
    return person_names


def get_person_last_names(cand):
    """
    Returns the last names for the two person mentions in candidate cand
    """
    person1_name, person2_name = get_person_text(cand)
    person1_lastname = (
        person1_name.split(" ")[-1] if len(person1_name.split(" ")) > 1 else None
    )
    person2_lastname = (
        person2_name.split(" ")[-1] if len(person2_name.split(" ")) > 1 else None
    )
    return person1_lastname, person2_lastname


@preprocessor
def get_text_between(cand: DataPoint) -> DataPoint:
    """
    Returns the text between the two person mentions in the sentence for a candidate
    """
    start = cand.person1_word_idx[1] + 1
    end = cand.person2_word_idx[0]
    cand.text_between = " ".join(cand.tokens[start:end])
    return cand


@preprocessor
def get_between_tokens(cand: DataPoint) -> DataPoint:
    """
    Returns the tokens between the two person mentions in the sentence for a candidate
    """
    start = cand.person1_word_idx[1] + 1
    end = cand.person2_word_idx[0]
    cand.between_tokens = cand.tokens[start:end]
    return cand


@preprocessor
def get_left_tokens(cand: DataPoint) -> DataPoint:
    """
    Returns the length window tokens between to the left of the person mentions
    """
    # TODO: need to pass window as input params
    window = 3

    end = cand.person1_word_idx[0]
    cand.person1_left_tokens = cand.tokens[0:end][-1 - window : -1]

    end = cand.person2_word_idx[0]
    cand.person2_left_tokens = cand.tokens[0:end][-1 - window : -1]
    return cand


@preprocessor
def get_right_tokens(cand: DataPoint) -> DataPoint:
    """
    Returns the length window tokens between to the right of the person mentions
    """
    # TODO: need to pass window as input params
    window = 3

    start = cand.person1_word_idx[1] + 1
    cand.person1_right_tokens = cand.tokens[start::][0 : window + 1]

    start = cand.person2_word_idx[1] + 1
    cand.person2_right_tokens = cand.tokens[start::][0 : window + 1]
    return cand
