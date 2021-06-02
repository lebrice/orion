from orion.benchmark.benchmark_client import get_or_create_benchmark

from orion.benchmark.assessment import AverageResult, AverageRank

from orion.benchmark.task import RosenBrock, EggHolder, CarromTable
from orion.benchmark.warm_start_study import WarmStartStudy
from orion.benchmark.assessment.warm_start_efficiency import WarmStartEfficiency
from orion.benchmark import Benchmark
from orion.benchmark.warm_start_benchmark import WarmStartBenchmark

from warmstart.new_knowledge_base import KnowledgeBase
from warmstart.tasks.quadratics import QuadraticsTask
from warmstart.tasks.profet import SvmTask
from ablr.ablr import ABLR
import numpy as np
from typing import List
# TODO: Need to make sure that in the "hot_start" case, all points have the same
# task id? Or maybe use the branching could be used?
from warmstart.distance import similarity, distance

# print(distance(task_a, task_b))
# print(similarity(task_a, task_b))

# print(task_a, task_b)
# exit()
target_task = QuadraticsTask(a0=0.1, a1=0.1, a2=0.1, task_id=0, max_trials=25)
N = 10

# BUG: Figure out why we observe '50' warm-start points, rather than 25.

benchmark = WarmStartBenchmark(
    name="warmstart_benchmark",
    algorithms=[
        # "random",
        "tpe",
        # "ablr",
        # ABLR,
        # "BayesianOptimizer", # Doesn't work!
    ],
    source_tasks = [
        target_task.get_similar_task(i * (1 / N), task_id=i, max_trials=25)
        for i in range(N+1)
    ],
    target_tasks = [
        target_task for _ in range(N+1)
    ],
    repetitions = 5,
    knowledge_base_type=KnowledgeBase,
    debug=True,
)





# benchmark = get_or_create_benchmark(
#     name="warmstart_benchmark",
#     algorithms=[
#         # "random",
#         "tpe",
#         # "ablr",
#         # ABLR,
#         # "BayesianOptimizer", # Doesn't work!
#     ],
#     targets=[
#         {
#             "assess": [WarmStartEfficiency(5)],
#             "source_tasks": [
#                 target_task.get_similar_task(i * (1 / N), task_id=i, max_trials=25)
#                 for i in range(N+1)
#             ],
#             "target_tasks": [
#                 target_task for _ in range(N+1)
#             ],
#         },
#     ],
#     knowledge_base_type=KnowledgeBase,
#     debug=True,
# )


benchmark.process()
status = benchmark.status(False)
figures = benchmark.analysis()
for figure in figures:
    figure.show()

# print(benchmark.studies)
# benchmark.studies[0].analysis()
# exps = benchmark.experiments(False)
