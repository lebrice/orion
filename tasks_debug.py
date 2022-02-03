from orion.benchmark.benchmark_client import get_or_create_benchmark

from orion.benchmark.assessment import AverageResult, AverageRank

from orion.benchmark.task import RosenBrock, EggHolder, CarromTable
from orion.benchmark.task.quadratics import QuadraticsTask
from orion.benchmark.task.profet import ProfetSvmTask, ProfetFcNetTask, ProfetForresterTask, ProfetXgBoostTask
from orion.benchmark.task.profet.profet_task import MetaModelTrainingConfig

import logging
from logging import getLogger as get_logger

# orion_logger = get_logger("orion")
# orion_logger.setLevel(logging.ERROR)

logger = get_logger("orion.benchmark.task")
logger.setLevel(logging.INFO)

# Configuration for the training of the meta-model used by the Profet algorithm.
train_config = MetaModelTrainingConfig(num_burnin_steps=10_000)

DEBUG = False
NAME = "task_debug"

benchmark = get_or_create_benchmark(
    name=NAME,
    algorithms=["random", "tpe", "robo_gp", "robo_ablr", "robo_dngo"],
    targets=[
        {
            "assess": [AverageResult(10)],
            "task": [
                # QuadraticsTask(25),
                # RosenBrock(25, dim=3),
                # EggHolder(20, dim=4),
                # CarromTable(20),
                ProfetSvmTask(max_trials=30, train_config=train_config),
                # FcNetTask(max_trials=30, train_config=train_config),
                # ForresterTask(max_trials=30, train_config=train_config),
                # XgBoostTask(max_trials=30, train_config=train_config),
            ],
        }
    ],
    storage={
        "type": "legacy",
        "database": {"type": "pickleddb", "host": f"{NAME}.pkl"},
    },
    debug=DEBUG,
)

benchmark.process(n_workers=1)
status=benchmark.status()
print(status)
from pathlib import Path

figures_dir = Path("figures") / benchmark.name
figures_dir.mkdir(parents=True, exist_ok=True)
figures = benchmark.analysis()
for i, figure in enumerate(figures):
    figure.write_image(str(figures_dir / f"warm_start_fig_{i:02}.png"))
    figure.write_html(
        str(figures_dir / f"warm_start_fig_{i:02}.html"), include_plotlyjs="cdn"
    )
    figure.show()