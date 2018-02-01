#!/usr/bin/env python


"""
TODO:
Currently, this test will fail because it tries to import things from Snorkel,
which is beyond the toplevel package. We will need to implement these tests
once we have separated Fonduer into it's own repo that imports from snorkel
as a module, rather than being embedded in snorkel directories itself.
"""

from tests.context import fonduer

def test_do_nothing():
    """This test does nothing."""
    assert True is True


