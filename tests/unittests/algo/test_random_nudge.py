from __future__ import annotations

from typing import Any, ClassVar
from orion.algo.base import BaseAlgorithm
from orion.testing.algo import BaseAlgoTests
from orion.algo.random_nudge import RandomNudge


class TestRandomNudge(BaseAlgoTests):
    algo_type: ClassVar[type[BaseAlgorithm]] = RandomNudge
    algo_name: ClassVar[str] = algo_type.__qualname__.lower()
    config: ClassVar[dict[str, Any]] = {"noise_variance": 0.1, "seed": None}
    max_trials: ClassVar[int] = 50
    space: ClassVar[dict[str, str]] = {
        "x": "uniform(0, 1)",
        "y": "uniform(0, 1)",
    }


TestRandomNudge.set_phases([("random", 0, "space.sample")])
