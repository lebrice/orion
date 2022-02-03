from typing import Dict, List, Mapping

import pytest

from orion.core.worker.multi_task_algo import AbstractKnowledgeBase, MultiTaskAlgo
from orion.core.worker.primary_algo import PrimaryAlgo
from orion.core.worker.trial import Trial
from tests.unittests.core.test_primary_algo import TestPrimaryAlgoWraps

# TODO: Reuse all the tests for the PrimaryAlgo for when the algo is warm-starteable?


@pytest.fixture()
def multi_task_algo_fn(dumbalgo, space, fixed_suggestion):
    """Set up a PrimaryAlgo with dumb configuration."""
    algo_config = {
        "DumbAlgo": {
            "value": fixed_suggestion,
            "subone": {"DumbAlgo": dict(value=6, scoring=5)},
        }
    }

    def get_algo(knowledge_base: AbstractKnowledgeBase) -> MultiTaskAlgo:
        return MultiTaskAlgo(space, algo_config, knowledge_base=knowledge_base)

    return get_algo


class DummyKnowledgeBase(AbstractKnowledgeBase):
    def get_reusable_trials(
        self, target_experiment, max_trials: int = None
    ) -> Dict[Mapping, List[Trial]]:
        return {}


@pytest.fixture()
def palgo(dumbalgo, space, fixed_suggestion):
    """Set up a PrimaryAlgo with dumb configuration."""
    algo_config = {
        "DumbAlgo": {
            "value": fixed_suggestion,
            "subone": {"DumbAlgo": dict(value=6, scoring=5)},
        }
    }
    palgo = MultiTaskAlgo(space, algo_config, kb=DummyKnowledgeBase())
    return palgo


class TestMultiTaskAlgoWrapper(TestPrimaryAlgoWraps):
    pass
