#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark definition
======================
"""
import copy
import itertools

from tabulate import tabulate

from orion.client import create_experiment

from .benchmark import Benchmark
from .study import Study
from .warm_start_benchmark import WarmStartBenchmark
from .warm_start_study import WarmStartStudy