from snorkel.contrib.babble import Explanation

from temp_image_class import BBox

# Box X = Person (category_id = 1)
# Box Y = Bike (category_id = 2)
A = BBox({'bbox': (100, 100, 100, 100), 'category_id': 1}, None)
B = BBox({'bbox': (150, 150, 100, 100), 'category_id': 2}, None)

"""
----------
| A       |
|     ----|-----
|    |    |    |
-----|----     |
     |       B |
     __________
"""

a_and_b = (A, B)

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

comparisons = [
    Explanation(
        condition="the top of box x is above the top of box y",
        label=True,
        candidate=a_and_b,
        semantics=None),
    Explanation(
        condition="the bottom of box y is below the bottom of box y",
        label=True,
        candidate=a_and_b,
        semantics=None),
]

    # Explanation(
    #     condition="bottom edge of Box X is below bottom edge of Box Y",
    #     label=False,
    #     candidate=11,
    #     semantics=None),        
    # # Natural 
    # Explanation(
    #     condition="Box X is very much larger than box Y.",
    #     label=False,
    #     candidate=11,
    #     semantics=None),

explanations = (edges + points + comparisons)