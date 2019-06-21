from snorkel.labeling.preprocess import preprocessor
from snorkel.types import FieldMap


def get_person_text(x, index):
    if index not in [0, 1]:
        ValueError("Invalid index. Index should be either 0 or 1.")

    field_name = "person{}_char_idx".format(index + 1)
    start = x[field_name][0]
    end = x[field_name][1]
    return x["sentence"][start : end + 1]


def get_person_last_names(x):
    person1_name = " ".join(get_person_text(x, 0).split(" "))
    person2_name = " ".join(get_person_text(x, 1).split(" "))

    person1_lastname = person1_name[-1] if len(person1_name) > 0 else None
    person2_lastname = person2_name[-1] if len(person2_name) > 0 else None

    return person1_lastname, person2_lastname


@preprocessor
def get_text_between(sentence, person1_char_idx, person2_char_idx) -> FieldMap:
    """
    Returns the text between the two person mentions in the sentence for a candidate
    """
    start = person1_char_idx[1] + 1
    end = person2_char_idx[0]
    return dict(text_between=sentence[start:end])


@preprocessor
def get_between_tokens(sentence, person1_char_idx, person2_char_idx) -> FieldMap:
    """
    Returns the tokens between the two person mentions in the sentence for a candidate
    """
    start = person1_char_idx[1] + 1
    end = person2_char_idx[0]
    tokens = sentence[start:end].split(" ")

    return dict(between_tokens=tokens)


@preprocessor
def get_person1_left_tokens(sentence, person1_char_idx, person2_char_idx) -> FieldMap:
    """
    Returns the length window tokens between to the left of the person_indexth person mentions
    """
    # TODO: need to pass person_idx and window as input params
    person_index = 0
    window = 3

    if person_index == 0:
        end = person1_char_idx[0]
        left_sent = sentence[0:end].split(" ")
    else:
        end = person2_char_idx[0]
        left_sent = sentence[0:end].split(" ")

    tokens = " ".join(left_sent).split(" ")[-1 - window : -1]  # removes extra spaces
    return dict(person1_left_tokens=tokens)


@preprocessor
def get_person2_left_tokens(sentence, person1_char_idx, person2_char_idx) -> FieldMap:
    """
    Returns the length window tokens between to the left of the person_indexth person mentions
    """
    # TODO: need to pass person_idx and window as input params
    person_index = 1
    window = 3

    if person_index == 0:
        end = person1_char_idx[0]
        left_sent = sentence[0:end].split(" ")
    else:
        end = person2_char_idx[0]
        left_sent = sentence[0:end].split(" ")

    tokens = " ".join(left_sent).split(" ")[-1 - window : -1]  # removes extra spaces
    return dict(person2_left_tokens=tokens)


@preprocessor
def get_person1_right_tokens(sentence, person1_char_idx, person2_char_idx) -> FieldMap:
    # TODO: need to pass person_idx and window as input params
    person_index = 0
    window = 3

    if person_index == 0:
        start = person1_char_idx[1] + 1
        right_sent = sentence[start::].split(" ")
    else:
        start = person2_char_idx[1] + 1
        right_sent = sentence[start::].split(" ")

    tokens = " ".join(right_sent).split(" ")[0 : window + 1]  # removes extra spaces
    return dict(person1_right_tokens=tokens)


@preprocessor
def get_person2_right_tokens(sentence, person1_char_idx, person2_char_idx) -> FieldMap:
    # TODO: need to pass person_idx and window as input params
    person_index = 0
    window = 3

    if person_index == 0:
        start = person1_char_idx[1] + 1
        right_sent = sentence[start::].split(" ")
    else:
        start = person2_char_idx[1] + 1
        right_sent = sentence[start::].split(" ")

    tokens = " ".join(right_sent).split(" ")[0 : window + 1]  # removes extra spaces
    return dict(person2_right_tokens=tokens)
