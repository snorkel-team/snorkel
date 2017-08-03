from snorkel.contrib.babble import Explanation

boxes = [
    # Bottom Edge
    Explanation(
        condition="bottom edge of Box 3 is below bottom edge of Box 5",
        label=False,
        candidate=11,
        semantics=None),
]

explanations = (boxes)