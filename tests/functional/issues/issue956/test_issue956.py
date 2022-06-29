import sys
from random import random

from orion.client import build_experiment, report_objective


def main():
    print(sys.argv[1:])
    #     orion hunt -c orion_bayesian_config.yaml \
    #   python 3 test_orion_2.py Breakout 4 "/some/path" "/some/other/path.yml" \
    #   --learning_rate~'loguniform(1e-6, 1e-3)' --alpha~'loguniform(0.94, 0.96)'
    report_objective(random())


def main_python_api():
    experiment = build_experiment(
        name="rc7",
        space={
            "learning_rate": "loguniform(1e-6, 1e-3)",
            "alpha": "loguniform(0.94, 0.96)",
        },
        algorithms=dict(
            tpe=dict(
                seed=None,
                n_initial_points=20,
                n_ei_candidates=25,
                gamma=0.25,
                equal_weight=False,
                prior_weight=1.0,
                full_weight_num=25,
                parallel_strategy=dict(
                    of_type="StatusBasedParallelStrategy",
                    strategy_configs=dict(broken=dict(of_type="MaxParallelStrategy")),
                    default_strategy=dict(of_type="NoParallelStrategy"),
                ),
            )
        ),
        debug=True,
    )
    while not experiment.is_done:
        trial = experiment.suggest()
        print(trial.params)
        experiment.observe(
            trial, [dict(name="objective", type="objective", value=random())]
        )


if __name__ == "__main__":
    # main()
    main_python_api()
