import unittest

import torch.nn as nn

from snorkel.classification.snorkel_classifier import Operation, Task

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
            Operation(
                name="the_first_layer", module_name="linear1", inputs=[("_input_", 0)]
            ),
            Operation(
                name="the_second_layer",
                module_name="linear2",
                inputs=[("the_first_layer", 0)],
            ),
        ]

        task = Task(name=TASK_NAME, module_pool=module_pool, task_flow=task_flow)

        # Task has no functionality on its own
        # Here we only confirm that the object was initialized
        self.assertEqual(task.name, TASK_NAME)


if __name__ == "__main__":
    unittest.main()
