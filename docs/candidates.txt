Candidates
==========

In order to apply machine learning--i.e., in this case, a classifier--to information extraction problems,
we need to have a base set of objects that are being classified.  In Snorkel, these are the `Candidate`
subclasses, which are defined over `Context` arguments, and represent *potential* mentions to extract.
We use `Matcher` operators to extract a set of `Candidate` objects from the input data.

Core Data Models
----------------

.. automodule:: snorkel.models.candidate
    :members:

Core Objects for Candidate Extraction
-------------------------------------

.. automodule:: snorkel.candidates
    :members:

.. automodule:: snorkel.matchers
    :members:
