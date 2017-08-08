from snorkel.contrib.babble import Explanation
from snorkel.contrib.babble.image import BBox

# Box X = Person (category_id = 1)
# Box Y = Bike (category_id = 2)
A = BBox({'bbox': (100, 100, 100, 100), 'category_id': 1}, None)
B = BBox({'bbox': (150, 150, 100, 100), 'category_id': 2}, None)
C = BBox({'bbox': (300, 300, 50, 50), 'category_id': 2}, None)

"""


        ----------
        | A       |
        |     ----|-----
        |    |    |    |
        -----|----     |
             |       B |
             __________


                             ----
                            | C  |
                             ----
"""

a_and_b = (A, B)
a_and_c = (A, C)


edges = [
    # Edges of same box
    Explanation(
        condition="the bottom of box x is below the top of box x",
        label=True,
        candidate=a_and_b,
        semantics=None),
    # Edges of different boxes
    Explanation(
        condition="the top of box y is below the top of box x",
        label=True,
        candidate=a_and_b,
        semantics=None),
]

points = [
    # Corner to Corner
    Explanation(
        condition="the bottom right corner of box x, is below the left top corner of box x",
        label=True,
        candidate=a_and_b,
        semantics=None),
    # Center to Center
    Explanation(
        condition="the center of box y is below the center of box x",
        label=True,
        candidate=a_and_b,
        semantics=None),
    # Center to Corner
    Explanation(
        condition="the center of box y is below the top right corner of box x",
        label=True,
        candidate=a_and_b,
        semantics=None),
    # Center to Edge
    Explanation(
        condition="the center of box y is below the top edge of box x",
        label=True,
        candidate=a_and_b,
        semantics=None),
]

boxes = [
    # Box above
    Explanation(
        condition="box x is above box y",
        label=True,
        candidate=a_and_b,
        semantics=None),
    # Box below
    Explanation(
        condition="box y is below box x",
        label=True,
        candidate=a_and_b,
        semantics=None),
    # Box left
    Explanation(
        condition="box x is left of box y",
        label=True,
        candidate=a_and_b,
        semantics=None),
    # Box right
    Explanation(
        condition="box y is right of box x",
        label=True,
        candidate=a_and_b,
        semantics=None),
    # Box near
    Explanation(
        condition="box x is near the top left corner of box y",
        label=True,
        candidate=a_and_b,
        semantics=None),
    # Box far
    Explanation(
        condition="box x is far away from box y",
        label=True,
        candidate=a_and_c,
        semantics=None),
    # Box within
    Explanation(
        condition="not box x is within box y", #TODO: FIX ME!
        label=True,
        candidate=a_and_b,
        semantics=None),
]

comparisons = [
    # Top
    Explanation(
        condition="the top of box x is above the top of box y",
        label=True,
        candidate=a_and_b,
        semantics=None),
    # Bottom
    Explanation(
        condition="the bottom of box y is below the bottom of box x",
        label=True,
        candidate=a_and_b,
        semantics=None),
    # Left
    Explanation(
        condition="the left of box x is to the left of the left edge of box y",
        label=True,
        candidate=a_and_b,
        semantics=None),
    # Right
    Explanation(
        condition="the right edge of box y is to the right of the left edge of box x",
        label=True,
        candidate=a_and_b,
        semantics=None),
    # Near
    Explanation(
        condition="the center of box x is near the top left corner of box y",
        label=True,
        candidate=a_and_b,
        semantics=None),
    # Far
    Explanation(
        condition="the top left corner of box x is far from the bottom right corner of box y",
        label=True,
        candidate=a_and_b,
        semantics=None),
]

quantified = [
    # Larger
    Explanation(
        condition="box x is larger than box y",
        label=True,
        candidate=a_and_c,
        semantics=None),
    # Much Larger
    # Explanation(
    #     condition="box x is much larger than box y",
    #     label=True,
    #     candidate=a_and_c,
    #     semantics=None),
    # Smaller
    Explanation(
        condition="box y is smaller than box x",
        label=True,
        candidate=a_and_c,
        semantics=None),
    # # Smaller
    # Explanation(
    #     condition="box y is much smaller than box x",
    #     label=True,
    #     candidate=a_and_c,
    #     semantics=None),
]

explanations = (edges + points + comparisons + boxes + quantified)