import unittest

from snorkel.model.utils import recursive_merge_dicts


class UtilsTest(unittest.TestCase):
    def test_recursive_merge_dicts(self):
        x = {"foo": {"Foo": {"FOO": 1}}, "bar": 2, "baz": 3}
        y = {"FOO": 4, "bar": 5}
        z = {"foo": 6}
        w = recursive_merge_dicts(x, y, verbose=False)
        self.assertEqual(w["bar"], 5)
        self.assertEqual(w["foo"]["Foo"]["FOO"], 4)
        with self.assertRaises(ValueError):
            recursive_merge_dicts(x, z, verbose=False)


if __name__ == "__main__":
    unittest.main()
