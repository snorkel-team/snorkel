import collections
import copy
import json
import os
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.optim as optim

from snorkel.classification import (
    DictDataLoader,
    DictDataset,
    MultitaskClassifier,
    Operation,
    Task,
    Trainer,
)
from snorkel.classification.training.loggers import LogWriter, TensorBoardWriter

TASK_NAMES = ["task1", "task2"]
base_config = {"n_epochs": 1, "progress_bar": False}
NUM_EXAMPLES = 6
BATCH_SIZE = 2
BATCHES_PER_EPOCH = NUM_EXAMPLES / BATCH_SIZE


def create_dataloader(task_name="task", split="train"):
    X = torch.FloatTensor([[i, i] for i in range(NUM_EXAMPLES)])
    Y = torch.ones(NUM_EXAMPLES).long()

    dataset = DictDataset(
        name="dataset", split=split, X_dict={"data": X}, Y_dict={task_name: Y}
    )

    dataloader = DictDataLoader(dataset, batch_size=BATCH_SIZE)
    return dataloader


def create_task(task_name, module_suffixes=("", "")):
    module1_name = f"linear1{module_suffixes[0]}"
    module2_name = f"linear2{module_suffixes[1]}"

    module_pool = nn.ModuleDict(
        {
            module1_name: nn.Sequential(nn.Linear(2, 10), nn.ReLU()),
            module2_name: nn.Linear(10, 2),
        }
    )

    op1 = Operation(module_name=module1_name, inputs=[("_input_", "data")])
    op2 = Operation(module_name=module2_name, inputs=[op1.name])

    op_sequence = [op1, op2]

    task = Task(name=task_name, module_pool=module_pool, op_sequence=op_sequence)

    return task


dataloaders = [create_dataloader(task_name) for task_name in TASK_NAMES]
tasks = [
    create_task(TASK_NAMES[0], module_suffixes=["A", "A"]),
    create_task(TASK_NAMES[1], module_suffixes=["A", "B"]),
]
model = MultitaskClassifier([tasks[0]])


class TrainerTest(unittest.TestCase):
    def test_trainer_onetask(self):
        """Train a single-task model"""
        trainer = Trainer(**base_config)
        trainer.fit(model, [dataloaders[0]])

    def test_trainer_twotask(self):
        """Train a model with overlapping modules and flows"""
        multitask_model = MultitaskClassifier(tasks)
        trainer = Trainer(**base_config)
        trainer.fit(multitask_model, dataloaders)

    def test_trainer_errors(self):
        dataloader = copy.deepcopy(dataloaders[0])

        # No train split
        trainer = Trainer(**base_config)
        dataloader.dataset.split = "valid"
        with self.assertRaisesRegex(ValueError, "Cannot find any dataloaders"):
            trainer.fit(model, [dataloader])

        # Unused split
        trainer = Trainer(**base_config, valid_split="val")
        with self.assertRaisesRegex(ValueError, "Dataloader splits must be"):
            trainer.fit(model, [dataloader])

    def test_checkpointer_init(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            more_config = {
                "checkpointing": True,
                "checkpointer_config": {"checkpoint_dir": temp_dir},
                "log_writer_config": {"log_dir": temp_dir},
            }
            trainer = Trainer(**base_config, **more_config, logging=True)
            trainer.fit(model, [dataloaders[0]])
            self.assertIsNotNone(trainer.checkpointer)

            broken_config = {
                "checkpointing": True,
                "checkpointer_config": {"checkpoint_dir": None},
                "log_writer_config": {"log_dir": temp_dir},
            }
            with self.assertRaises(TypeError):
                trainer = Trainer(**base_config, **broken_config, logging=False)
                trainer.fit(model, [dataloaders[0]])

    def test_log_writer_init(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_writer_config = {"log_dir": temp_dir}
            trainer = Trainer(
                **base_config,
                logging=True,
                log_writer="json",
                log_writer_config=log_writer_config,
            )
            trainer.fit(model, [dataloaders[0]])
            self.assertIsInstance(trainer.log_writer, LogWriter)

            log_writer_config = {"log_dir": temp_dir}
            trainer = Trainer(
                **base_config,
                logging=True,
                log_writer="tensorboard",
                log_writer_config=log_writer_config,
            )
            trainer.fit(model, [dataloaders[0]])
            self.assertIsInstance(trainer.log_writer, TensorBoardWriter)

            log_writer_config = {"log_dir": temp_dir}
            with self.assertRaisesRegex(ValueError, "Unrecognized writer"):
                trainer = Trainer(
                    **base_config,
                    logging=True,
                    log_writer="foo",
                    log_writer_config=log_writer_config,
                )
                trainer.fit(model, [dataloaders[0]])

    def test_log_writer_json(self):
        # Addresses issue #1439
        # Confirm that a log file is written to the specified location after training
        run_name = "log.json"
        with tempfile.TemporaryDirectory() as temp_dir:
            log_writer_config = {"log_dir": temp_dir, "run_name": run_name}
            trainer = Trainer(
                **base_config,
                logging=True,
                log_writer="json",
                log_writer_config=log_writer_config,
            )
            trainer.fit(model, [dataloaders[0]])
            log_path = os.path.join(trainer.log_writer.log_dir, run_name)
            with open(log_path, "r") as f:
                log = json.load(f)
            self.assertIn("model/all/train/loss", log)

    def test_optimizer_init(self):
        trainer = Trainer(**base_config, optimizer="sgd")
        trainer.fit(model, [dataloaders[0]])
        self.assertIsInstance(trainer.optimizer, optim.SGD)

        trainer = Trainer(**base_config, optimizer="adam")
        trainer.fit(model, [dataloaders[0]])
        self.assertIsInstance(trainer.optimizer, optim.Adam)

        trainer = Trainer(**base_config, optimizer="adamax")
        trainer.fit(model, [dataloaders[0]])
        self.assertIsInstance(trainer.optimizer, optim.Adamax)

        with self.assertRaisesRegex(ValueError, "Unrecognized optimizer"):
            trainer = Trainer(**base_config, optimizer="foo")
            trainer.fit(model, [dataloaders[0]])

    def test_scheduler_init(self):
        trainer = Trainer(**base_config, lr_scheduler="constant")
        trainer.fit(model, [dataloaders[0]])
        self.assertIsNone(trainer.lr_scheduler)

        trainer = Trainer(**base_config, lr_scheduler="linear")
        trainer.fit(model, [dataloaders[0]])
        self.assertIsInstance(trainer.lr_scheduler, optim.lr_scheduler.LambdaLR)

        trainer = Trainer(**base_config, lr_scheduler="exponential")
        trainer.fit(model, [dataloaders[0]])
        self.assertIsInstance(trainer.lr_scheduler, optim.lr_scheduler.ExponentialLR)

        trainer = Trainer(**base_config, lr_scheduler="step")
        trainer.fit(model, [dataloaders[0]])
        self.assertIsInstance(trainer.lr_scheduler, optim.lr_scheduler.StepLR)

        with self.assertRaisesRegex(ValueError, "Unrecognized lr scheduler"):
            trainer = Trainer(**base_config, lr_scheduler="foo")
            trainer.fit(model, [dataloaders[0]])

    def test_warmup(self):
        lr_scheduler_config = {"warmup_steps": 1, "warmup_unit": "batches"}
        trainer = Trainer(**base_config, lr_scheduler_config=lr_scheduler_config)
        trainer.fit(model, [dataloaders[0]])
        self.assertEqual(trainer.warmup_steps, 1)

        lr_scheduler_config = {"warmup_steps": 1, "warmup_unit": "epochs"}
        trainer = Trainer(**base_config, lr_scheduler_config=lr_scheduler_config)
        trainer.fit(model, [dataloaders[0]])
        self.assertEqual(trainer.warmup_steps, BATCHES_PER_EPOCH)

        lr_scheduler_config = {"warmup_percentage": 1 / BATCHES_PER_EPOCH}
        trainer = Trainer(**base_config, lr_scheduler_config=lr_scheduler_config)
        trainer.fit(model, [dataloaders[0]])
        self.assertEqual(trainer.warmup_steps, 1)

    def test_save_load(self):
        non_base_config = {"n_epochs": 2, "progress_bar": False}
        trainer1 = Trainer(**base_config, lr_scheduler="exponential")
        trainer1.fit(model, [dataloaders[0]])
        trainer2 = Trainer(**non_base_config, lr_scheduler="linear")
        trainer3 = Trainer(**non_base_config, lr_scheduler="linear")

        with tempfile.NamedTemporaryFile() as fd:
            checkpoint_path = fd.name
            trainer1.save(checkpoint_path)
            trainer2.load(checkpoint_path, model=model)
            trainer3.load(checkpoint_path, None)

        self.assertEqual(trainer1.config, trainer2.config)
        self.dict_check(
            trainer1.optimizer.state_dict(), trainer2.optimizer.state_dict()
        )

        # continue training after load
        trainer2.fit(model, [dataloaders[0]])

        # check that an inappropriate model does not load an optimizer state but a trainer config
        self.assertEqual(trainer1.config, trainer3.config)
        self.assertFalse(hasattr(trainer3, "optimizer"))
        trainer3.fit(model, [dataloaders[0]])

    def dict_check(self, dict1, dict2):
        for k in dict1.keys():
            dict1_ = dict1[k]
            dict2_ = dict2[k]
            if isinstance(dict1_, collections.Mapping):
                self.dict_check(dict1_, dict2_)
            elif isinstance(dict1_, torch.Tensor):
                self.assertTrue(
                    torch.eq(
                        dict1_,
                        dict2_,
                    ).all()
                )
            else:
                self.assertEqual(dict1_, dict2_)


if __name__ == "__main__":
    unittest.main()
