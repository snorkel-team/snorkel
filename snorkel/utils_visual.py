from .models import Phrase, TemporarySpan
from collections import namedtuple


Bbox = namedtuple('bbox', ['page', 'top', 'bottom', 'left', 'right'], verbose=False)


def bbox_from_span(span):
    if isinstance(span, TemporarySpan) and span.is_visual():
        return Bbox(
                span.get_attrib_tokens('page')[0],
                min(span.get_attrib_tokens('top')),
                max(span.get_attrib_tokens('bottom')),
                min(span.get_attrib_tokens('left')),
                max(span.get_attrib_tokens('right')))
    else:
        return None


def bbox_from_phrase(phrase):
    # TODO: this may have issues where a phrase is linked to words on different pages
    if isinstance(phrase, Phrase) and phrase.is_visual():
        return Bbox(
                phrase.page[0],
                min(phrase.top),
                max(phrase.bottom),
                min(phrase.left),
                max(phrase.right))
    else:
        return None


def bbox_horz_aligned(box1, box2):
    """
    Returns true if the vertical center point of either span is within the 
    vertical range of the other
    """
    if not (box1 and box2): return False
    center1 = (box1.bottom + box1.top) / 2.0
    center2 = (box2.bottom + box2.top) / 2.0
    return ((center1 >= box2.top and center1 <= box2.bottom) or
            (center2 >= box1.top and center2 <= box1.bottom))


def bbox_vert_aligned(box1, box2):
    """
    Returns true if the horizontal center point of either span is within the 
    horizontal range of the other
    """
    if not (box1 and box2): return False
    center1 = (box1.right + box1.left) / 2.0
    center2 = (box2.right + box2.left) / 2.0
    return ((center1 >= box2.left and center1 <= box2.right) or
            (center2 >= box1.left and center2 <= box1.right))


def bbox_vert_aligned_left(box1, box2):
    """
    Returns true if the left boundary of both boxes is within 2 pts
    """
    if not (box1 and box2): return False
    return abs(box1.left - box2.left) <= 2


def bbox_vert_aligned_right(box1, box2):
    """
    Returns true if the right boundary of both boxes is within 2 pts
    """
    if not (box1 and box2): return False
    return abs(box1.right - box2.right) <= 2


def bbox_vert_aligned_center(box1, box2):
    """
    Returns true if the right boundary of both boxes is within 5 pts
    """
    if not (box1 and box2): return False
    return abs((box1.right + box1.left) / 2.0 - (box2.right + box2.left) / 2.0) <= 5
