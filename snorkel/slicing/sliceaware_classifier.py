from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from snorkel.analysis import Scorer
from snorkel.classification import DictDataLoader, DictDataset, Operation, Task
from snorkel.classification.data import DEFAULT_INPUT_DATA_KEY, DEFAULT_TASK_NAME
from snorkel.classification.multitask_classifier import MultitaskClassifier

from .utils import add_slice_labels, convert_to_slice_tasks


class SliceAwareClassifier(MultitaskClassifier):
    """A slice-aware classifier that supports training + scoring on slice labels.

    NOTE: This model currently only supports binary classification.

    Parameters
    ----------
    base_architecture
        A network architecture that accepts input data and outputs a representation
    head_dim
        Output feature dimension of the base_architecture, and input dimension of the
        internal prediction head: ``nn.Linear(head_dim, 2)``.
    slice_names
        A list of slice names that the model will accept initialize as tasks
        and accept as corresponding labels
    scorer
        A Scorer to be used for initialization of the ``MultitaskClassifier`` superclass.
    **multitask_kwargs
        Arbitrary keyword arguments to be passed to the ``MultitaskClassifier`` superclass.

    Attributes
    ----------
    base_task
        A base ``snorkel.classification.Task`` that the model will learn.
        This becomes a ``master_head_module`` that combines slice tasks information.
        For more, see ``snorkel.slicing.convert_to_slice_tasks``.
    slice_names
        See above
    """

    def __init__(
        self,
        base_architecture: nn.Module,
        head_dim: int,
        slice_names: List[str],
        input_data_key: str = DEFAULT_INPUT_DATA_KEY,
        task_name: str = DEFAULT_TASK_NAME,
        scorer: Scorer = Scorer(metrics=["accuracy", "f1"]),
        **multitask_kwargs: Any,
    ) -> None:

        # Initialize module_pool with 1) base_architecture and 2) prediction_head
        # Assuming `head_dim` can be used to map base_architecture to prediction_head
        module_pool = nn.ModuleDict(
            {
                "base_architecture": base_architecture,
                "prediction_head": nn.Linear(head_dim, 2),
            }
        )

        # Create op_sequence from base_architecture -> prediction_head
        op_sequence = [
            Operation(
                name="input_op",
                module_name="base_architecture",
                inputs=[("_input_", input_data_key)],
            ),
            Operation(
                name="head_op", module_name="prediction_head", inputs=["input_op"]
            ),
        ]

        # Initialize base_task using specified base_architecture
        self.base_task = Task(
            name=task_name,
            module_pool=module_pool,
            op_sequence=op_sequence,
            scorer=scorer,
        )

        # Convert base_task to associated slice_tasks
        slice_tasks = convert_to_slice_tasks(self.base_task, slice_names)

        # Initialize a MultitaskClassifier with all slice_tasks
        model_name = f"{task_name}_sliceaware_classifier"
        super().__init__(tasks=slice_tasks, name=model_name, **multitask_kwargs)
        self.slice_names = slice_names

    def make_slice_dataloader(
        self, dataset: DictDataset, S: np.recarray, **dataloader_kwargs: Any
    ) -> DictDataLoader:
        """Create DictDataLoader with slice labels, initialized from specified dataset.

        Parameters
        ----------
        dataset
            A DictDataset that will be converted into a slice-aware dataloader
        S
            A [num_examples, num_slices] slice matrix indicating whether
            each example is in every slice
        slice_names
            A list of slice names corresponding to columns of ``S``

        dataloader_kwargs
            Arbitrary kwargs to be passed to DictDataLoader
            See ``DictDataLoader.__init__``.
        """

        # Base task must have corresponding labels in dataset
        if self.base_task.name not in dataset.Y_dict:  # type: ignore
            raise ValueError(
                f"Base task ({self.base_task.name}) labels missing from {dataset}"
            )

        # Initialize dataloader
        dataloader = DictDataLoader(dataset, **dataloader_kwargs)

        # Make dataloader slice-aware
        add_slice_labels(dataloader, self.base_task, S)

        return dataloader

    @torch.no_grad()
    def score_slices(
        self, dataloaders: List[DictDataLoader], as_dataframe: bool = False
    ) -> Union[Dict[str, float], pd.DataFrame]:
        """Scores appropriate slice labels using the overall prediction head.

        In other words, uses ``base_task`` (NOT ``slice_tasks``) to evaluate slices.

        In practice, we'd like to use a final prediction from a _single_ task head.
        To do so, ``self.base_task`` leverages reweighted slice representation to
        make a prediction. In this method, we remap all slice-specific ``pred``
        labels to ``self.base_task`` for evaluation.

        Parameters
        ----------
        dataloaders
            A list of DictDataLoaders to calculate scores for
        as_dataframe
            A boolean indicating whether to return results as pandas
            DataFrame (True) or dict (False)
        eval_slices_on_base_task
            A boolean indicating whether to remap slice labels to base task.
            Otherwise, keeps evaluation of slice labels on slice-specific heads.

        Returns
        -------
        Dict[str, float]
            A dictionary mapping metric¡ names to corresponding scores
            Metric names will be of the form "task/dataset/split/metric"
        """

        eval_mapping: Dict[str, Optional[str]] = {}
        # Collect all labels
        all_labels: Union[List, Set] = []
        for dl in dataloaders:
            all_labels.extend(dl.dataset.Y_dict.keys())  # type: ignore
        all_labels = set(all_labels)

        # By convention, evaluate on "pred" labels, not "ind" labels
        # See ``snorkel.slicing.utils.add_slice_labels`` for more about label creation
        for label in all_labels:
            if "pred" in label:
                eval_mapping[label] = self.base_task.name
            elif "ind" in label:
                eval_mapping[label] = None

        # Call score on the original remapped set of labels
        return super().score(
            dataloaders=dataloaders,
            remap_labels=eval_mapping,
            as_dataframe=as_dataframe,
        )
