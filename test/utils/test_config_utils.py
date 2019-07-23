import unittest

from snorkel.types import Config
from snorkel.utils.config_utils import merge_config


class FooConfig(Config):
    a: float = 0.5


class BarConfig(Config):
    a: int = 1
    foo_config: FooConfig = FooConfig()  # type: ignore


class UtilsTest(unittest.TestCase):
    def test_merge_config(self):
        config_updates = {"a": 2, "foo_config": {"a": 0.75}}
        bar_config = merge_config(BarConfig(), config_updates)
        self.assertEqual(bar_config.a, 2)
        self.assertEqual(bar_config.foo_config.a, 0.75)
