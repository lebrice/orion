#!/usr/bin/env python
"""Simple one dimensional example for a possible user's script."""
import argparse

import yaml

from orion.client import report_results


def function(x):
    """Evaluate partial information of a quadratic."""
    z = x - 34.56789
    return 4 * z**2 + 23.4, 8 * z


def execute():
    """Execute a simple pipeline as an example."""
    # 1. Receive inputs as you want
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", required=True)
    inputs = parser.parse_args()

    with open(inputs.configuration) as f:
        config = yaml.safe_load(f)

    # 2. Perform computations

    y, dy = function(config["x"])

    # 3. Gather and report results
    results = list()
    results.append(dict(name="example_objective", type="objective", value=y))
    results.append(dict(name="example_gradient", type="gradient", value=[dy]))

    report_results(results)


if __name__ == "__main__":
    execute()
