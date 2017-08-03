from snorkel.contrib.babble import Explanation

boxes = {
    # Name
    Explanation(
        condition="True",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
}

explanations = (boxes)