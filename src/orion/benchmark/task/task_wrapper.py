from abc import ABC
from typing import (
    Any,
    Dict,
    Generic,
    List,
    TypeVar,
    Union,
)

from orion.benchmark.task.base import BenchmarkTask, BenchmarkTask
from logging import getLogger as get_logger



logger = get_logger(__name__)


TaskType = TypeVar("TaskType", bound=BenchmarkTask)


class TaskWrapper(BenchmarkTask, Generic[TaskType], ABC):
    """ ABC for a Wrapper around a Task.

    Wrappers could be used to modify the search space, trials, or the results of the task.
    
    Parameters
    ----------
    task : TaskType
        A task to wrap.
    max_trials : int, optional
        Maximum number of trials. By default None, in which case the max trials is the same as that
        of the wrapped task.
    """

    def __init__(self, task: TaskType, max_trials: int = None):
        # TODO: Should Wrappers pass the `task` as an argument to the super() constructor?
        # When/How are these objects deserialized?
        if isinstance(task, dict):
            if len(task) != 1:
                raise ValueError(
                    f"Expected task configuration dict to have only one key (the task name), but "
                    f"got {task} instead."
                )
            task_name, task_config = task.popitem()
            from orion.benchmark.task.base import bench_task_factory
            task = bench_task_factory.create(of_type=task_name, **task_config)
        super().__init__(max_trials=max_trials or task.max_trials)
        self.task: TaskType = task

    def call(self, **kwargs) -> List[Dict]:
        return self.task.call(**kwargs)

    def get_search_space(self) -> Dict[str, str]:
        return self.task.get_search_space()

    @property
    def unwrapped(self) -> Union[TaskType, BenchmarkTask]:
        """ Returns the 'unwrapped' task. """
        return self.task.unwrapped

    @property
    def configuration(self) -> Dict[str, Any]:
        """Return the configuration of the task wrapper (including the wrapped task's configuration).
        """
        return {
            type(self).__qualname__: {
                "task": self.task.configuration,
                "max_trials": self.max_trials,
            }
        }

