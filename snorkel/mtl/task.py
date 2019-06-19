import logging

from torch import nn


class Task(object):
    """A single Task in a multi-task problem

    :param name: The name of the task (Primary key).
    :type name: str
    :param module_pool: A dict of all modules that uses in the task.
    :type module_pool: nn.ModuleDict
    :param task_flow: The task flow among modules to define how the data flows.
    :type task_flow: list
    :param loss_func: The function to calculate the loss.
    :type loss_func: function
    :param output_func: The function to generate the output.
    :type output_func: function
    :param scorer: The class of metrics to evaluate the task.
    :type scorer: Scorer class
    """

    def __init__(self, name, module_pool, task_flow, loss_func, output_func, scorer):
        self.name = name
        assert isinstance(module_pool, nn.ModuleDict) is True
        self.module_pool = module_pool
        self.task_flow = task_flow
        self.loss_func = loss_func
        self.output_func = output_func
        self.scorer = scorer

        logging.info(f"Created task: {self.name}")

    def __repr__(self):
        cls_name = type(self).__name__
        return f"{cls_name}(name={self.name})"
