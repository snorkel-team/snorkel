from .sequential_scheduler import SequentialScheduler
from .shuffled_scheduler import ShuffledScheduler

batch_schedulers = {"sequential": SequentialScheduler, "shuffled": ShuffledScheduler}
