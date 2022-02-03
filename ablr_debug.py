import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Type

import numpy as np
import pandas as pd
import plotly
import logging
from simple_parsing import ArgumentParser
from logging import getLogger as get_logger
logger = get_logger("simple_parsing")
logger.setLevel(logging.ERROR)

# from ablr.ablr import ABLR
# from warmstart.distance import distance, similarity
# from warmstart.new_knowledge_base import KnowledgeBase
from orion.benchmark.task.profet import ProfetSvmTask
from orion.benchmark.task.quadratics import QuadraticsTask

import orion
from orion.algo.robo.bohamiann import OrionBohamiannWrapper
from orion.benchmark import Benchmark
from orion.benchmark.assessment import AverageRank, AverageResult
from orion.benchmark.assessment.warm_start_efficiency import WarmStartEfficiency
from orion.benchmark.assessment.warm_start_task_correlation import (
    warm_start_task_correlation_figure,
)

# Useless imports below this, just to try and get the algos to be found
from orion.benchmark.benchmark_client import get_or_create_benchmark
from orion.benchmark.task import CarromTable, EggHolder, RosenBrock, Branin
from orion.benchmark.warm_start_benchmark import (
    WarmStartBenchmark,
)
from orion.benchmark.warm_start_study import WarmStartStudy

N_REPEATS = 1
# SEED = 123
N_TRIALS_PER_TASK = 25
NAME="mtablr_debug"
DEBUG = True

# BUG: #629 (https://github.com/Epistimio/orion/issues/629)
from orion.core.worker.primary_algo import PrimaryAlgo
from orion.testing.space import build_space
_ = PrimaryAlgo(space=build_space(), algorithm_config="random") 


"""
Update page: https://hackmd.io/2BHiHPJ1SgOhXy-X16NSNw
"""
benchmark = get_or_create_benchmark(
    name=NAME,
    algorithms=[
        # TODO: Weird bug! If you don't put a 'built-in' algo as the first item, it doesn't work!
        # "tpe",  # Buggy: fail to sample in [0, 1]
        # "random",
        # "robo_gp",
        "robo_ablr",
        "robo_mtablr",  # TODO: Not ready yet.
        # "robo_dngo",
        # "robo_gp_mcmc",
        # "robo_bohamiann",
    ],
    targets=[
        {
            "assess": [AverageResult(N_REPEATS)], #AverageRank(N_REPEATS)],
            "task": [
                Branin(max_trials=N_TRIALS_PER_TASK),
                # RosenBrock(max_trials=N_TRIALS_PER_TASK),
                # EggHolder(max_trials=N_TRIALS_PER_TASK),
                # CarromTable(max_trials=N_TRIALS_PER_TASK),
            ],
        }
    ],
    storage={
        "type": "legacy",
        "database": {"type": "pickleddb", "host": f"{NAME}.pkl"},
    },
    debug=DEBUG,
)

benchmark.setup_studies()
benchmark.process(n_workers=1)
figures = benchmark.analysis()
for figure in figures:
    figure.show()
status = benchmark.status(silent=False)
print(status)