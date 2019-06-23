from snorkel.labeling.preprocess import preprocessor
from snorkel.types import FieldMap


def get_person_text(x):
    """
    Returns the text for the two person mentions in candidate x
    """
    person_names = []
    for index in [1, 2]:
        field_name = "person{}_word_idx".format(index)
        start = x[field_name][0]
        end = x[field_name][1] + 1
        person_names.append(' '.join(x["tokens"][start:end]))
    return person_names


def get_person_last_names(x):
    person1_name, person2_name = get_person_text(x)
    person1_lastname = person1_name.split(' ')[-1] if len(person1_name.split(' ')) > 0 else None
    person2_lastname = person2_name.split(' ')[-1] if len(person2_name.split(' ')) > 0 else None
    return person1_lastname, person2_lastname


def strip_chars(tokens):
    return [word.lstrip().strip("'") for word in tokens]

@preprocessor
def get_text_between(tokens, person1_word_idx, person2_word_idx) -> FieldMap:
    """
    Returns the text between the two person mentions in the sentence for a candidate
    """
    start = person1_word_idx[1] + 1
    end = person2_word_idx[0]
    text_between = ' '.join(tokens[start:end])
    return dict(text_between=text_between)


@preprocessor
def get_between_tokens(tokens, person1_word_idx, person2_word_idx) -> FieldMap:
    """
    Returns the tokens between the two person mentions in the sentence for a candidate
    """
    start = person1_word_idx[1] + 1
    end = person2_word_idx[0]

    return dict(between_tokens=strip_chars(tokens[start:end]))


@preprocessor
def get_left_tokens(tokens, person1_word_idx, person2_word_idx) -> FieldMap:
    """
    Returns the length window tokens between to the left of the person mentions
    """
    # TODO: need to pass window as input params
    window = 3

    for index in [1, 2]:
        end = person1_word_idx[0]
        person1_left_tokens = strip_chars(tokens[0:end][-1 - window : -1])

        end = person2_word_idx[0]
        person2_left_tokens = strip_chars(tokens[0:end][-1 - window : -1])
    return dict(person1_left_tokens=person1_left_tokens, person2_left_tokens=person2_left_tokens)


@preprocessor
def get_right_tokens(tokens, person1_word_idx, person2_word_idx) -> FieldMap:
    """
    Returns the length window tokens between to the right of the person mentions
    """
    # TODO: need to pass window as input params
    window = 3

    for index in [1, 2]:
        start = person1_word_idx[1] + 1
        person1_right_tokens = strip_chars(tokens[start::][0:window + 1])

        start = person2_word_idx[1] + 1
        person2_right_tokens = strip_chars(tokens[start::][0:window + 1])
    return dict(person1_right_tokens=person1_right_tokens, person2_right_tokens=person2_right_tokens)