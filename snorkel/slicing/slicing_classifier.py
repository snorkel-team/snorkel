from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import torch
import torch.nn as nn

from snorkel.analysis import Scorer
from snorkel.classification import DictDataLoader, DictDataset, Operation, Task
from snorkel.classification.multitask_classifier import MultitaskClassifier

from .utils import add_slice_labels, convert_to_slice_tasks


class BinarySlicingClassifier(MultitaskClassifier):
    """A slice-aware binary classifier that supports training + scoring on slice labels.

    Parameters
    ----------
    representation_net
        A representation network architecture that accepts input data and
        outputs a representation of ``head_dim``
    head_dim
        Output feature dimension of the representation_net, and input dimension of the
        internal prediction head: ``nn.Linear(head_dim, 2)``.
    slice_names
        A list of slice names that the model will accept as tasks/labels
    scorer
        A Scorer to be used for initialization of the ``BaseMultitaskClassifier`` superclass.
    **multitask_kwargs
        Arbitrary key-word arguments to be passed to the ``BaseMultitaskClassifier`` superclass.

    Attributes
    ----------
    base_input_name
        A naming convention for input data field key required in DictDataset.X_dict
    base_task_name
        A naming convention for label_name key required in DictDataset.Y_dict
    base_task
    slice_names
        See above
    """

    def __init__(
        self,
        representation_net: nn.Module,
        head_dim: int,
        slice_names: List[str],
        input_data_key: str,
        task_name: str,
        scorer: Scorer = Scorer(metrics=["accuracy", "f1"]),
        **multitask_kwargs: Any,
    ) -> None:

        module_pool = nn.ModuleDict(
            {
                "representation_net": representation_net,
                # By convention, initialize binary classification as 2-dim output
                "prediction_head": nn.Linear(head_dim, 2),
            }
        )

        op_sequence = [
            Operation(
                name="input_op",
                module_name="representation_net",
                inputs=[("_input_", input_data_key)],
            ),
            Operation(
                name="head_op", module_name="prediction_head", inputs=["input_op"]
            ),
        ]

        self.base_task = Task(
            name=task_name,
            module_pool=module_pool,
            op_sequence=op_sequence,
            scorer=scorer,
        )

        slice_tasks = convert_to_slice_tasks(self.base_task, slice_names)

        # Initialize a MultitaskClassifier under the hood
        model_name = f"{task_name}_slicing_classifier"
        super().__init__(tasks=slice_tasks, name=model_name, **multitask_kwargs)
        self.slice_names = slice_names

    def make_slice_dataloader(
        self, dataset: DictDataset, S: np.ndarray, **dataloader_kwargs: Any
    ) -> DictDataLoader:
        """Create DictDataLoaders with slice labels for initialized slice tasks.

        Parameters
        ----------
        datasets
            A list of DictDataset
        S
        slice_names

        dataloader_kwargs
            Arbitrary kwargs to be passed to DictDataloader
        """
        if S.shape[1] != len(self.slice_names):
            raise ValueError("Num columns in S matrix does not match num slice_names.")

        if self.base_task.name not in dataset.Y_dict:  # type: ignore
            raise ValueError(
                f"Base task ({self.base_task.name}) labels missing from {dataset}"
            )

        dataloader = DictDataLoader(dataset, **dataloader_kwargs)
        add_slice_labels(dataloader, self.base_task, S, self.slice_names)

        return dataloader

    @torch.no_grad()
    def score_slices(
        self, dataloaders: List[DictDataLoader], as_dataframe: bool = False
    ) -> Dict[str, float]:
        """Create label mapping from appropriate slice labels to base task, and calls _score.

        For more, see ``BaseMultitaskClassifier._score``.

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

        return super().score(
            dataloaders=dataloaders,
            remap_labels=eval_mapping,
            as_dataframe=as_dataframe,
        )
