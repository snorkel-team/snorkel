from snorkel.models import Bbox
from snorkel.contrib.babble import Explanation

# Box X = Person (category_id = 1)
# Box Y = Bike (category_id = 2)

A = Bbox(top=100, bottom=200, left=100, right=200)
B = Bbox(top=150, bottom=250, left=150, right=250)
C = Bbox(top=300, bottom=350, left=300, right=350)
D = Bbox(top=250, bottom=450, left=250, right=450)

"""


        ----------
        | A       |
        |     ----|-----
        |    |    |    |
        -----|----     |
             |       B |
             __________-----------------
                       | D              |
                       |    ----        |
                       |   | C  |       |
                       |    ----        |
                       |                |
                       |                |
                       -----------------
      
"""

a_and_b = (A, B)
a_and_c = (A, C)
d_and_c = (D, C)


edges = [
    # Edges of same box
    Explanation(
        condition="the bottom of box x is below the top of box x",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.below', ('.edge', ('.box', ('.int', 0)), ('.string', 'top'))), ('.edge', ('.box', ('.int', 0)), ('.string', 'bottom')))))),
    # Edges of different boxes
    Explanation(
        condition="the top of box y is below the top of box x",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.below', ('.edge', ('.box', ('.int', 0)), ('.string', 'top'))), ('.edge', ('.box', ('.int', 1)), ('.string', 'top')))))),
]

points = [
    # Corner to Corner
    Explanation(
        condition="the bottom right corner of box x, is below the left top corner of box x",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.below', ('.corner', ('.box', ('.int', 0)), ('.string', 'top'), ('.string', 'left'))), ('.corner', ('.box', ('.int', 0)), ('.string', 'right'), ('.string', 'bottom')))))),
    # Center to Center
    Explanation(
        condition="the center of box y is below the center of box x",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.below', ('.center', ('.box', ('.int', 0)))), ('.center', ('.box', ('.int', 1))))))),
    # Center to Corner
    Explanation(
        condition="the center of box y is below the top right corner of box x",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.below', ('.corner', ('.box', ('.int', 0)), ('.string', 'right'), ('.string', 'top'))), ('.center', ('.box', ('.int', 1))))))),
    # Center to Edge
    Explanation(
        condition="the center of box y is below the top edge of box x",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.below', ('.edge', ('.box', ('.int', 0)), ('.string', 'top'))), ('.center', ('.box', ('.int', 1))))))),
]

boxes = [
    # Box above
    Explanation(
        condition="box x is above box y",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.above', ('.box', ('.int', 1))), ('.box', ('.int', 0)))))),
    # Box below
    Explanation(
        condition="box y is below box x",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.below', ('.box', ('.int', 0))), ('.box', ('.int', 1)))))),
    # Box left
    Explanation(
        condition="box x is left of box y",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.left', ('.box', ('.int', 1))), ('.box', ('.int', 0)))))),
    # Box right
    Explanation(
        condition="box y is to the right of box x",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.right', ('.box', ('.int', 0))), ('.box', ('.int', 1)))))),
    # Box near (point and box)
    Explanation(
        condition="box x is near the top left corner of box y",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.near', ('.corner', ('.box', ('.int', 1)), ('.string', 'left'), ('.string', 'top'))), ('.box', ('.int', 0)))))),
    # Box near (box and box)
    Explanation(
        condition="box x is in the same place as box y",
        label=True,
        candidate=d_and_c,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.near', ('.box', ('.int', 1))), ('.box', ('.int', 0)))))),
    # Box far
    Explanation(
        condition="box x is far away from box y",
        label=True,
        candidate=a_and_c,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.far', ('.box', ('.int', 1))), ('.box', ('.int', 0)))))),
    # Box within
    Explanation(
        condition="box x is not within box y",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.not', ('.call', ('.within', ('.box', ('.int', 1))), ('.box', ('.int', 0))))))),
]

comparisons = [
    # Top
    Explanation(
        condition="the top of box x is above the top of box y",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.above', ('.edge', ('.box', ('.int', 1)), ('.string', 'top'))), ('.edge', ('.box', ('.int', 0)), ('.string', 'top')))))),
    # Bottom
    Explanation(
        condition="the bottom of box y is below the bottom of box x",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.below', ('.edge', ('.box', ('.int', 0)), ('.string', 'bottom'))), ('.edge', ('.box', ('.int', 1)), ('.string', 'bottom')))))),
    # Past (bottom)
    Explanation(
        condition="box y is past the bottom of box x",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.below', ('.box', ('.int', 0))), ('.box', ('.int', 1)))))),
    # Left
    Explanation(
        condition="the left of box x is to the left of the left edge of box y",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.left', ('.edge', ('.box', ('.int', 1)), ('.string', 'left'))), ('.edge', ('.box', ('.int', 0)), ('.string', 'left')))))),
    # Right
    Explanation(
        condition="the right edge of box y is to the right of the left edge of box x",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.right', ('.edge', ('.box', ('.int', 0)), ('.string', 'left'))), ('.edge', ('.box', ('.int', 1)), ('.string', 'right')))))),
    # Near
    Explanation(
        condition="the center of box x is near the top left corner of box y",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.near', ('.corner', ('.box', ('.int', 1)), ('.string', 'left'), ('.string', 'top'))), ('.center', ('.box', ('.int', 0))))))),
    # Far
    Explanation(
        condition="the top left corner of box x is far from the bottom right corner of box y",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.far', ('.corner', ('.box', ('.int', 1)), ('.string', 'right'), ('.string', 'bottom'))), ('.corner', ('.box', ('.int', 0)), ('.string', 'left'), ('.string', 'top')))))),
]

quantified = [
    # Larger
    Explanation(
        condition="box x is larger than box y",
        label=True,
        candidate=a_and_c,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.larger', ('.box', ('.int', 1))), ('.box', ('.int', 0)))))),
    # Much Larger
    # Explanation(
    #     condition="box x is much larger than box y",
    #     label=True,
    #     candidate=a_and_c,
    #     semantics=None),
    # Smaller
    Explanation(
        condition="box Y is smaller than box X",
        label=True,
        candidate=a_and_c,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.smaller', ('.box', ('.int', 0))), ('.box', ('.int', 1)))))),
    # Same Area
    Explanation(
        condition="box X is the same size as box Y",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.samearea', ('.box', ('.int', 1))), ('.box', ('.int', 0)))))),
    # # Smaller
    # Explanation(
    #     condition="box y is much smaller than box x",
    #     label=True,
    #     candidate=a_and_c,
    #     semantics=None),
    # Taller
    Explanation(
        condition="box x is taller than box y",
        label=True,
        candidate=a_and_c,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.taller', ('.box', ('.int', 1))), ('.box', ('.int', 0)))))),
    # Wider
    Explanation(
        condition=" Box X is wider than Box Y",
        label=True,
        candidate=a_and_c,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.wider', ('.box', ('.int', 1))), ('.box', ('.int', 0)))))),
    # Same Width
    Explanation(
        condition="box X is as wide as box Y",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.samewidth', ('.box', ('.int', 1))), ('.box', ('.int', 0)))))),
    # Overlaps (0.25 thresh)
    Explanation(
        condition="box x overlaps with box y",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.overlaps', ('.box', ('.int', 1))), ('.box', ('.int', 0)))))),
    # Within
    Explanation(
        condition="box y is within with box x",
        label=True,
        candidate=d_and_c,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.within', ('.box', ('.int', 0))), ('.box', ('.int', 1)))))),
    # Surrounds
    Explanation(
        condition="box x surrounds box y",
        label=True,
        candidate=d_and_c,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.surrounds', ('.box', ('.int', 1))), ('.box', ('.int', 0)))))),
]

parser = [
    #Period at the end of the sentence, explanation 121
    Explanation(
        condition="Box X is much wider than Box Y.",
        label=True,
        candidate=a_and_c,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.wider', ('.box', ('.int', 1))), ('.box', ('.int', 0)))))),
        #Period at the end of the sentence, explanation 121
    Explanation(
        condition="Box Y is to the right of Box X.",
        label=True,
        candidate=a_and_c,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.right', ('.box', ('.int', 0))), ('.box', ('.int', 1)))))),
]

possessives = [
    # Bbox's side
    Explanation(
        condition="Box x's top edge is above box y's center.",
        label=True,
        candidate=a_and_c,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.above', ('.center', ('.box', ('.int', 1)))), ('.edge', ('.box', ('.int', 0)), ('.string', 'top')))))),
    # Bbox's center
    Explanation(
        condition="the center of box y is below box x's center",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.below', ('.center', ('.box', ('.int', 0)))), ('.center', ('.box', ('.int', 1))))))),
    # Bbox's corner
    Explanation(
        condition="the bottom right corner of box x, is below box x's left top corner",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.below', ('.corner', ('.box', ('.int', 0)), ('.string', 'top'), ('.string', 'left'))), ('.corner', ('.box', ('.int', 0)), ('.string', 'right'), ('.string', 'bottom')))))),
    # Bbox's left/right
    Explanation(
        condition="box y is to box x's right",
        label=True,
        candidate=a_and_b,
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.right', ('.box', ('.int', 0))), ('.box', ('.int', 1)))))),
]

explanations = (edges + points + comparisons + boxes + quantified + parser + possessives)