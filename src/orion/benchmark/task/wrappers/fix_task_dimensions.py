""" Wrapper around a task that fixes dimensions of it's space to the given values. """
from typing import Dict, Any, List
from orion.benchmark.task.task_wrapper import TaskWrapper, TaskType
from orion.algo.space import Space
from orion.core.io.space_builder import SpaceBuilder


class FixTaskDimensionsWrapper(TaskWrapper[TaskType]):
    """ Wrapper around a Task that fixes the values of some of its input dimensions.

    The value returned by `get_search_space` is also modified.

    Parameters
    ----------
    task : TaskType
        A task to wrap.
    fixed_dims : Dict[str, Any]
        Dictionary mapping from the name of a dimension of the task space to the value that
        dimension should be fixed at.
    max_trials : int, optional
        Maximum number of trials, by default None
    """

    def __init__(self, task: TaskType, fixed_dims: Dict[str, Any], max_trials: int = None):
        super().__init__(task=task, max_trials=max_trials)
        self.fixed_dims = fixed_dims
        # The whole space, from the wrapped task.
        self._full_space: Space = SpaceBuilder().build(self.task.get_search_space())

        for dimension_name in fixed_dims:
            if dimension_name not in self._full_space:
                raise ValueError(
                    f"Can't fix dimension '{dimension_name}' because it isn't in the wrapped "
                    f"task's space: {self._full_space}."
                )

        # The part of the space that can be modified.
        self._space: Space = SpaceBuilder().build(self.get_search_space())

    def call(self, **kwargs) -> List[Dict]:
        """ Calls the wrapped task, passing the kwargs with some values fixed. """
        new_kwargs = kwargs.copy()
        new_kwargs.update(self.fixed_dims)
        return super().call(**new_kwargs)

    def get_search_space(self) -> Dict[str, str]:
        """ Returns the truncated search space (without the fixed values). """
        original_space = super().get_search_space()
        modified_space = original_space.copy()
        for dimension_name in self.fixed_dims.keys():
            modified_space.pop(dimension_name)
        return modified_space

    @property
    def configuration(self) -> Dict[str, Any]:
        """Return the configuration of the task wrapper.
        """
        config = super().configuration
        config[type(self).__qualname__]["fixed_dims"] = self.fixed_dims.copy()
        return config
