from snorkel.contrib.babble import Explanation

from temp_image_class import BBox

# Box X = Person (category_id = 1)
# Box Y = Bike (category_id = 2)
A = BBox({'bbox': (100, 100, 50, 25), 'category_id': 1}, None)
B = BBox({'bbox': (100, 125, 50, 75), 'category_id': 2}, None)

x_below_y = (B, A)

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
        candidate=x_below_y,
        semantics=None),
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
]

explanations = (boxes)