from __future__ import annotations

import sys
from typing import Any, ClassVar

if sys.version_info < (3, 8):
    from typing_extensions import Literal
else:
    from typing import Literal

import numpy as np

from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space
from orion.core.utils import format_trials
from orion.core.worker.trial import Trial


class RandomNudge(BaseAlgorithm):
    """Example algo, where we add noise to the best trial observed so far and suggest that.

    Potential improvements:
    - Add a new parameter that sets a minimum number of random guesses to take before starting the
      greedy phase.
    - Use a different noise value for each dimension of the search-space, maybe based on the
      variance of the previous best trials in that dimension.
    - Reduce the "temperature" of the random guesses over time as more trials are observed
      (i.e. the noise variance).
    - Select the 'base' from the top-K best best trials, instead of only the best trial.
    - When identifying a new "best trial", keep track of the difference between it and the previous,
      use this as the "gradient", and do something cool with it, e.g. add a "momentum" term.
    """

    requires_type: ClassVar[Literal["real", "integer", "numerical", None]] = "real"
    requires_shape: ClassVar[Literal["flattened", None]] = "flattened"
    requires_dist: ClassVar[Literal["linear", None]] = "linear"

    def __init__(
        self, space: Space, noise_variance: float = 0.1, seed: int | None = None
    ):
        super().__init__(space)
        self.space: Space = space
        self.noise_variance = noise_variance
        self.seed = seed

        self.best_trial: Trial | None = None
        self.best_objective: float | None = None
        self.rng = np.random.RandomState(seed)

    def seed_rng(self, seed: int | None):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def observe(self, trials: list[Trial]) -> None:
        for trial in trials:
            if not self.has_observed(trial):
                self.register(trial)

            if trial.objective is not None:
                # Update the best trial seen so far.
                if self.best_trial is None:
                    self.best_trial = trial
                    self.best_objective = trial.objective.value
                elif trial.objective.value < self.best_objective:
                    self.best_trial = trial
                    self.best_objective = trial.objective.value

    def suggest(self, num: int) -> list[Trial]:
        if self.best_trial is None:
            # Haven't observed a trial with results yet, so just make random guesses.
            return self.space.sample(num, seed=self.rng)

        # Use the best trial observed so far as a base.
        best_trial_array = np.array(
            format_trials.trial_to_tuple(self.best_trial, space=self.space)
        )
        bounds_list: list[tuple[float, float]] = self.space.interval()
        lower_bound, upper_bound = zip(*bounds_list)

        trials: list[Trial] = []
        for index in range(num):
            noisy_guess = self.rng.normal(
                loc=best_trial_array,
                scale=self.noise_variance,
            )
            # Make sure the trial doesn't go out of the bounds.
            trial_array = np.clip(noisy_guess, a_min=lower_bound, a_max=upper_bound)
            trial = format_trials.tuple_to_trial(trial_array, space=self.space)

            # Register the trial (important for the state of the algo to be saved properly).
            self.register(trial)
            trials.append(trial)

        return trials

    @property
    def state_dict(self) -> dict:
        state: dict[str, Any] = super().state_dict
        state.update(
            best_trial=self.best_trial, best_objective=self.best_objective, rng=self.rng
        )
        return state

    def set_state(self, state_dict: dict) -> None:
        super().set_state(state_dict)
        self.best_trial = state_dict["best_trial"]
        self.best_objective = state_dict["best_objective"]
        self.rng = state_dict["rng"]

    @property
    def configuration(self) -> dict:
        self._param_names = ["noise_variance", "seed"]
        return super().configuration
