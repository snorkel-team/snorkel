import unittest
from functools import partial

import torch.nn as nn

from snorkel.classification.models.advanced import Task
from snorkel.classification.models.advanced.utils import ce_loss, softmax
from snorkel.classification.scorer import Scorer

TASK_NAME = "TestTask"


class TaskTest(unittest.TestCase):
    def test_task_creation(self):
        module_pool = nn.ModuleDict(
            {
                "linear1": nn.Sequential(nn.Linear(2, 10), nn.ReLU()),
                "linear2": nn.Linear(10, 1),
            }
        )

        task_flow = [
            {
                "name": "the_first_layer",
                "module": "linear1",
                "inputs": [("_input_", 0)],
            },
            {
                "name": "the_second_layer",
                "module": "linear2",
                "inputs": [("the_first_layer", 0)],
            },
        ]

        task = Task(
            name=TASK_NAME,
            module_pool=module_pool,
            task_flow=task_flow,
            loss_func=partial(ce_loss, "the_second_layer"),
            output_func=partial(softmax, "the_second_layer"),
            scorer=Scorer(metrics=["accuracy"]),
        )

        # Task has no functionality on its own
        # Here we only confirm that the object was initialized
        self.assertEqual(task.name, TASK_NAME)


if __name__ == "__main__":
    unittest.main()
