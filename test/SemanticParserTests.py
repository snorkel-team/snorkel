import unittest


class TestSemanticParsing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_compile_with_deps(self):
        # Make candidates (or fake candidates)

        # Confirm the generated LFs are executable
        self.assertEqual(1,1)
        self.assertFalse(False)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
