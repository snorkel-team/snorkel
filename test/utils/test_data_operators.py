import unittest

from snorkel.utils.data_operators import check_unique_names


class DataOperatorsTest(unittest.TestCase):
    def test_check_unique_names(self):
        check_unique_names(["alice", "bob", "chuck"])
        with self.assertRaisesRegex(ValueError, "3 operators with name c"):
            check_unique_names(["a", "a", "b", "c", "c", "c"])
