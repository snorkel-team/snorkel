from snorkel.contrib.babble import Explanation

# Box X = Person
# Box Y = Bike

boxes = [
    # Boxes
    Explanation(
        condition="box X is not box Y",
        label=True,
        candidate=11,
        semantics=None),
    # Edges
    Explanation(
        condition="the top of box x is not the bottom of box y",
        label=True,
        candidate=11,
        semantics=None),
    Explanation(
        condition="the left of box x is not the right of box y",
        label=True,
        candidate=11,
        semantics=None),
    # Edge Comparison
    Explanation(
        condition="bottom edge of Box X is below bottom edge of Box Y",
        label=False,
        candidate=11,
        semantics=None),
    Explanation(
        condition="bottom edge of Box X is below bottom edge of Box Y",
        label=False,
        candidate=11,
        semantics=None),        
    # # Natural 
    # Explanation(
    #     condition="Box X is very much larger than box Y.",
    #     label=False,
    #     candidate=11,
    #     semantics=None),
]

explanations = (boxes)