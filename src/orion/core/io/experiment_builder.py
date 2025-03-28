# pylint:disable=too-many-lines
"""
Create experiment from user options
===================================

Functions which build :class:`orion.core.worker.experiment.Experiment` objects based on user
configuration.

The instantiation of an :class:`orion.core.worker.experiment.Experiment` is not a trivial process
when the user request an experiment with specific options. One can easily create a new experiment
with ``Experiment('some_experiment_name')``, but the configuration of a _writable_ experiment is
less straightforward. This is because there is many sources of configuration and they have a strict
hierarchy. From the more global to the more specific, there is:

1. Global configuration:

  Defined by ``orion.core.DEF_CONFIG_FILES_PATHS``.
  Can be scattered in user file system, defaults could look like:

    - ``/some/path/to/.virtualenvs/orion/share/orion.core``
    - ``/etc/xdg/xdg-ubuntu/orion.core``
    - ``/home/${USER}/.config/orion.core``

  Note that most variables have default value even if user do not defined them in global
  configuration. These are defined in ``orion.core.__init__``.

2. Oríon specific environment variables:

   Environment variables which can override global configuration

    - Database specific:

      * ``ORION_DB_NAME``
      * ``ORION_DB_TYPE``
      * ``ORION_DB_ADDRESS``

3. Experiment configuration inside the database

  Configuration of the experiment if present in the database.
  Making this part of the configuration of the experiment makes it possible
  for the user to execute an experiment by only specifying partial configuration. The rest of the
  configuration is fetched from the database.

  For example, a user could:

    1. Rerun the same experiment

      Only providing the name is sufficient to rebuild the entire configuration of the
      experiment.

    2. Make a modification to an existing experiment

      The user can provide the name of the existing experiment and only provide the changes to
      apply on it. Here is an minimal example where we fully initialize a first experiment with a
      config file and then branch from it with minimal information.

      .. code-block:: bash

          # Initialize root experiment
          orion hunt --init-only --config previous_exeriment.yaml ./userscript -x~'uniform(0, 10)'
          # Branch a new experiment
          orion hunt -n previous_experiment ./userscript -x~'uniform(0, 100)'

4. Configuration file

  This configuration file is meant to overwrite the configuration coming from the database.
  If this configuration file was interpreted as part of the global configuration, a user could
  only modify an experiment using command line arguments.

5. Command-line arguments

  Those are the arguments provided to ``orion`` for any method (hunt, insert, etc). It includes the
  argument to ``orion`` itself as well as the user's script name and its arguments.

"""
from __future__ import annotations

import copy
import datetime
import getpass
import logging
import pprint
import sys
import typing
from typing import Any, TypeVar

import orion.core
from orion.algo.base import BaseAlgorithm, algo_factory
from orion.algo.space import Space
from orion.core.evc.adapters import BaseAdapter
from orion.core.evc.conflicts import ExperimentNameConflict, detect_conflicts
from orion.core.io import resolve_config
from orion.core.io.config import ConfigurationError
from orion.core.io.database import DuplicateKeyError
from orion.core.io.experiment_branch_builder import ExperimentBranchBuilder
from orion.core.io.interactive_commands.branching_prompt import BranchingPrompt
from orion.core.io.space_builder import SpaceBuilder
from orion.core.utils import backward
from orion.core.utils.exceptions import (
    BranchingEvent,
    NoConfigurationError,
    NoNameError,
    RaceCondition,
)
from orion.core.worker.experiment import Experiment, Mode
from orion.core.worker.experiment_config import ExperimentConfig
from orion.core.worker.primary_algo import create_algo
from orion.core.worker.warm_start import KnowledgeBase
from orion.storage.base import setup_storage

if typing.TYPE_CHECKING:
    from orion.core.evc.adapters import CompositeAdapter
    from orion.storage.base import BaseStorageProtocol
log = logging.getLogger(__name__)


##
# Functions to build experiments
##


def clean_config(name: str, config: dict, branching: dict | None):
    """Clean configuration from hidden fields (ex: ``_id``) and update branching if necessary"""
    log.debug("Cleaning config")

    config = copy.deepcopy(config)
    for key, value in list(config.items()):
        if key.startswith("_") or value is None:
            log.debug(f"Ignoring field {key}")
            config.pop(key)

    # TODO: Remove for v0.4
    if "strategy" in config:
        config["producer"] = {"strategy": config.pop("strategy")}

    if branching is None:
        branching = {}

    if branching.get("branch_from"):
        branching.setdefault("branch_to", name)
        name = branching["branch_from"]

    log.debug("Cleaned experiment config")
    log.debug("    Experiment config:\n%s", pprint.pformat(config))
    log.debug("    Branching config:\n%s", pprint.pformat(branching))

    return name, config, branching


def merge_algorithm_config(config: dict, new_config: dict) -> None:
    """Merge given algorithm configuration with db config"""
    # TODO: Find a better solution
    if isinstance(config.get("algorithm"), dict) and len(config["algorithm"]) > 1:
        log.debug("Overriding algo config with new one.")
        log.debug("    Old config:\n%s", pprint.pformat(config["algorithm"]))
        log.debug("    New config:\n%s", pprint.pformat(new_config["algorithm"]))
        config["algorithm"] = new_config["algorithm"]


# TODO: Remove for v0.4
def merge_producer_config(config: dict, new_config: dict) -> None:
    """Merge given producer configuration with db config"""
    if (
        isinstance(config.get("producer", {}).get("strategy"), dict)
        and len(config["producer"]["strategy"]) > 1
    ):
        log.debug("Overriding strategy config with new one.")
        log.debug("    Old config:\n%s", pprint.pformat(config["producer"]["strategy"]))
        log.debug(
            "    New config:\n%s", pprint.pformat(new_config["producer"]["strategy"])
        )

        config["producer"]["strategy"] = new_config["producer"]["strategy"]


##
# Private helper functions to build experiments
##


def _instantiate_adapters(config: list[dict]) -> CompositeAdapter:
    """Instantiate the adapter object

    Parameters
    ----------
    config: list
         List of adapter configurations to build a CompositeAdapter for the EVC.

    """
    return BaseAdapter.build(config)


def _instantiate_space(config: Space | dict[str, Any]) -> Space:
    """Instantiate the space object

    Build the Space object if argument is a dictionary, else return the Space object as is.

    Parameters
    ----------
    config: dict or Space object
        Dictionary of priors or already built Space object.

    """
    if isinstance(config, Space):
        return config

    return SpaceBuilder().build(config)


def _instantiate_knowledge_base(kb_config: dict[str, Any]) -> KnowledgeBase:
    """Instantiate the Knowledge base from its configuration."""
    if len(kb_config) != 1:
        raise ConfigurationError(
            f"The configuration for the KB should only have one key (the name of the KB "
            f"class) and the dict of kwargs. (got {kb_config})"
        )
    kb_type_name = list(kb_config.keys())[0]
    kb_types_with_name = [
        subclass
        for subclass in (KnowledgeBase.__subclasses__() + [KnowledgeBase])
        if subclass.__name__ == kb_type_name
    ]
    if len(kb_types_with_name) == 0:
        raise ConfigurationError(
            f"Unable to find a subclass of KnowledgeBase with the given name: {kb_type_name}"
        )
    if len(kb_types_with_name) > 1:
        raise ConfigurationError(
            f"Multiple subclasses of KnowledgeBase with the given name: {kb_type_name}"
        )
    kb_type = kb_types_with_name[0]
    kb_kwargs = kb_config[kb_type_name]
    # Instantiate the storage that is required for the KB.
    storage_config = kb_kwargs["storage"]
    if isinstance(storage_config, dict):
        storage = setup_storage(storage_config)
        kb_kwargs["storage"] = storage
    return kb_type(**kb_kwargs)


def _instantiate_algo(
    space: Space,
    max_trials: int | None,
    config: type[BaseAlgorithm] | str | dict | None = None,
    ignore_unavailable: bool = False,
    knowledge_base: KnowledgeBase | None = None,
):
    """Instantiate the algorithm object

    Parameters
    ----------
    config:
        Configuration of the algorithm. If None or empty, system's defaults are used
        (orion.core.config.experiment.algorithm).
    ignore_unavailable: bool, optional
        If True and algorithm is not available (plugin not installed), return the configuration.
        Otherwise, raise Factory error.

    """
    config = config or orion.core.config.experiment.algorithm
    assert config is not None
    try:
        algo_type: type[BaseAlgorithm]
        algo_config: dict
        if isinstance(config, str):
            algo_type = algo_factory.get_class(config)
            algo_config = {}
        elif isinstance(config, dict):
            backported_config = backward.port_algo_config(config)
            algo_name = backported_config.pop("of_type")
            algo_type = algo_factory.get_class(algo_name)
            algo_config = backported_config
        else:
            assert issubclass(config, BaseAlgorithm)
            algo_type = config
            algo_config = {}

        wrapped_algo = create_algo(
            space=space,
            algo_type=algo_type,
            knowledge_base=knowledge_base,
            **algo_config,
        )
        if max_trials is not None:
            wrapped_algo.max_trials = max_trials

    except NotImplementedError as e:
        if not ignore_unavailable:
            raise e
        log.warning(str(e))
        log.warning("Algorithm will not be instantiated.")
        wrapped_algo = config

    return wrapped_algo


def _instantiate_strategy(config: dict | None = None) -> None:
    """Instantiate the strategy object

    Parameters
    ----------
    config: dict, optional
        Configuration of the strategy. If None of empty, system's defaults are used
        (orion.core.config.producer.strategy).

    """
    if config or orion.core.config.experiment.strategy != {}:
        log.warning(
            "`strategy` option is not supported anymore. It should be set in "
            "algorithm configuration directly."
        )

    return None


def _fetch_config_version(
    configs: list[ExperimentConfig], version: int | None = None
) -> ExperimentConfig:
    """Fetch the experiment configuration corresponding to the given version

    Parameters
    ----------
    configs: list
        List of configurations fetched from storoge.
    version: int, optional
        Version to select. If None, last version will be selected. If version given is larger than
        largest version available, the largest version will be selected.

    """
    max_version = max(configs, key=lambda exp: exp.get("version", 1)).get("version", 1)

    if version is None:
        version = max_version
    else:
        version = version

    if version > max_version:
        log.warning(
            "Version %s was specified but most recent version is only %s. Using %s.",
            version,
            max_version,
            max_version,
        )

    version = min(version, max_version)

    filtered_configs = filter(lambda exp: exp.get("version", 1) == version, configs)

    return next(iter(filtered_configs))


###
# Functions for commandline API
###


def get_cmd_config(cmdargs) -> ExperimentConfig:
    """Fetch configuration defined by commandline and local configuration file.

    Arguments of commandline have priority over options in configuration file.
    """
    cmdargs = resolve_config.fetch_config_from_cmdargs(cmdargs)

    cmd_config = resolve_config.fetch_config(cmdargs)
    cmd_config = resolve_config.merge_configs(cmd_config, cmdargs)

    cmd_config.update(cmd_config.pop("experiment", {}))
    cmd_config["user_script_config"] = cmd_config.get("worker", {}).get(
        "user_script_config", None
    )

    cmd_config["branching"] = cmd_config.pop("evc", {})

    # TODO: We should move branching specific stuff below in a centralized place for EVC stuff.
    if (
        cmd_config["branching"].get("auto_resolution", False)
        and cmdargs.get("manual_resolution", None) is None
    ):
        cmd_config["branching"]["manual_resolution"] = False

    non_monitored_arguments = cmdargs.get("non_monitored_arguments")
    if non_monitored_arguments:
        cmd_config["branching"][
            "non_monitored_arguments"
        ] = non_monitored_arguments.split(":")

    # TODO: user_args won't be defined if reading from DB only (`orion hunt -n <exp> ` alone)
    metadata = resolve_config.fetch_metadata(
        cmd_config.get("user"),
        cmd_config.get("user_args"),
        cmd_config.get("user_script_config"),
    )
    cmd_config["metadata"] = metadata
    cmd_config.pop("config", None)

    cmd_config["space"] = cmd_config["metadata"].get("priors", None)

    backward.update_db_config(cmd_config)
    return cmd_config


def build_from_args(cmdargs):
    """Build an experiment based on commandline arguments.

    Options provided in commandline and configuration file (--config) will overwrite system's
    default values and configuration from database if experiment already exits.
    Commandline arguments have precedence over configuration file options.

    .. seealso::

        :func:`orion.core.io.experiment_builder.build` for more information on experiment creation.

    """

    cmd_config = get_cmd_config(cmdargs)

    # breakpoint()
    if "name" not in cmd_config:
        raise NoNameError()

    builder = ExperimentBuilder(cmd_config["storage"], debug=cmd_config.get("debug"))

    return builder.build(**cmd_config)


def get_from_args(cmdargs, mode="r"):
    """Build an experiment view based on commandline arguments

    .. seealso::

        :func:`orion.core.io.experiment_builder.load` for more information on creation of read-only
        experiments.

    """
    cmd_config = get_cmd_config(cmdargs)

    if "name" not in cmd_config:
        raise NoNameError()

    builder = ExperimentBuilder(cmd_config["storage"], debug=cmd_config.get("debug"))

    name = cmd_config.get("name")
    version = cmd_config.get("version")

    return builder.load(name, version, mode=mode)


def build(
    name: str,
    version: int | None = None,
    branching: dict | None = None,
    storage: BaseStorageProtocol | dict | None = None,
    **config,
):
    """Build an experiment.

    .. seealso::

        :func:`orion.core.io.experiment_builder.Experiment.build` for more information

    """
    if storage is None:
        storage = setup_storage()

    return ExperimentBuilder(storage).build(name, version, branching, **config)


def load(name, version=None, mode="r", storage=None):
    """Load an experiment.

    .. seealso::

        :func:`orion.core.io.experiment_builder.Experiment.load` for more information

    """
    if storage is None:
        storage = setup_storage()
    return ExperimentBuilder(storage).load(name, version, mode)


class ExperimentBuilder:
    """Utility to make new experiments using the same storage object.

    Parameters
    ----------
    storage: dict or BaseStorageProtocol, optional
        Storage object or storage configuration.
    debug: bool, optional.
        If True, force using EphemeralDB for the storage. Default: False
    """

    def __init__(
        self, storage: dict | BaseStorageProtocol | None = None, debug: bool = False
    ) -> None:
        singleton = None
        log.debug("Using for storage %s", storage)

        if not isinstance(storage, dict):
            singleton = storage
            storage = None

        if singleton is None:
            if storage is None:
                log.debug("Setting up storage from default config")

            self.storage_config = storage
            self.storage = setup_storage(storage, debug=debug)
        else:
            self.storage = singleton

    def build(
        self,
        name: str,
        version: int | None = None,
        branching: dict | None = None,
        **config,
    ) -> Experiment:
        """Build an experiment object

        If new, ``space`` argument must be provided, else all arguments are fetched from the
        database based on (name, version). If any argument given does not match the corresponding
        ones in the database for given (name, version), than the version is incremented and the
        experiment will be a child of the previous version.

        Parameters
        ----------
        name: str
            Name of the experiment to build
        version: int, optional
            Version to select. If None, last version will be selected.
            If version given is larger than largest version available, the largest version
            will be selected.
        space: dict, optional
            Optimization space of the algorithm.
            Should have the form ``dict(name='<prior>(args)')``.
        algorithm: str or dict, optional
            Algorithm used for optimization.
        strategy: str or dict, optional
            Deprecated and will be remove in v0.4. It should now be set in algorithm configuration
            directly if it supports it.
        max_trials: int, optional
            Maximum number of trials before the experiment is considered done.
        max_broken: int, optional
            Number of broken trials for the experiment to be considered broken.
        branching: dict, optional
            Arguments to control the branching.

            branch_from: str, optional
                Name of the experiment to branch from.
            manual_resolution: bool, optional
                Starts the prompt to resolve manually the conflicts. Defaults to False.
            non_monitored_arguments: list of str, optional
                Will ignore these arguments while looking for differences. Defaults to [].
            ignore_code_changes: bool, optional
                Will ignore code changes while looking for differences. Defaults to False.
            algorithm_change: bool, optional
                Whether to automatically solve the algorithm conflict (change of algo config).
                Defaults to True.
            orion_version_change: bool, optional
                Whether to automatically solve the orion version conflict.
                Defaults to True.
            code_change_type: str, optional
                How to resolve code change automatically. Must be one of 'noeffect', 'unsure' or
                'break'.  Defaults to 'break'.
            cli_change_type: str, optional
                How to resolve cli change automatically.
                Must be one of 'noeffect', 'unsure' or 'break'.
                Defaults to 'break'.
            config_change_type: str, optional
                How to resolve config change automatically. Must be one of 'noeffect', 'unsure' or
                'break'.  Defaults to 'break'.

        """
        log.debug(f"Building experiment {name} with {version}")
        log.debug("    Passed experiment config:\n%s", pprint.pformat(config))
        log.debug("    Branching config:\n%s", pprint.pformat(branching))

        name, config, branching = clean_config(name, config, branching)

        config = self.consolidate_config(name, version, config)

        if "space" not in config:
            raise NoConfigurationError(
                f"Experiment {name} does not exist in DB and space was not defined."
            )

        if len(config["space"]) == 0:
            raise NoConfigurationError("No prior found. Please include at least one.")

        experiment = self.create_experiment(mode="x", **copy.deepcopy(config))
        if experiment.id is None:
            log.debug("Experiment not found in DB. Now attempting registration in DB.")
            try:
                self._register_experiment(experiment)
                log.debug("Experiment successfully registered in DB.")
            except DuplicateKeyError:
                log.debug(
                    "Experiment registration failed. This is likely due to a race condition. "
                    "Now rolling back and re-attempting building it."
                )
                experiment = self.build(branching=branching, **config)

            return experiment

        log.debug(f"Experiment {config['name']}-v{config['version']} already existed.")

        conflicts = self._get_conflicts(experiment, branching)
        must_branch = len(conflicts.get()) > 1 or branching.get("branch_to")

        if must_branch and branching.get("enable", orion.core.config.evc.enable):
            return self._attempt_branching(conflicts, experiment, version, branching)
        elif must_branch:
            log.warning(
                "Running experiment in a different state:\n%s",
                self._get_branching_status_string(conflicts, branching),
            )

        log.debug("No branching required.")

        self._update_experiment(experiment)
        return experiment

    def _get_conflicts(self, experiment: Experiment, branching: dict):
        """Get conflicts between current experiment and corresponding configuration in database"""
        log.debug("Looking for conflicts in new configuration.")
        db_experiment = self.load(experiment.name, experiment.version, mode="r")
        conflicts = detect_conflicts(
            db_experiment.configuration, experiment.configuration, branching
        )

        log.debug(f"{len(conflicts.get())} conflicts detected:\n {conflicts.get()}")

        # elif must_branch and not enable_branching:
        #     raise ValueError("Configuration is different and generate a branching event")

        return conflicts

    def load(self, name: str, version: int | None = None, mode: Mode = "r"):
        """Load experiment from database

        An experiment view provides all reading operations of standard experiment but prevents the
        modification of the experiment and its trials.

        Parameters
        ----------
        name: str
            Name of the experiment to build
        version: int, optional
            Version to select. If None, last version will be selected.
            If version given is larger than largest version available,
            the largest version will be selected.
        mode: str, optional
            The access rights of the experiment on the database.
            'r': read access only
            'w': can read and write to database
            Default is 'r'

        """
        assert mode in set("rw")

        log.debug(
            f"Loading experiment {name} (version={version}) from database in mode `{mode}`"
        )
        db_config = self.fetch_config_from_db(name, version)

        if not db_config:
            version = version if version else "*"
            message = (
                f"No experiment with given name '{name}' and version '{version}' inside database, "
                "no view can be created."
            )
            raise NoConfigurationError(message)

        db_config.setdefault("version", 1)

        return self.create_experiment(mode=mode, **db_config)

    def fetch_config_from_db(self, name: str, version: int | None = None):
        """Fetch configuration from database

        Parameters
        ----------
        name: str
            Name of the experiment to fetch
        version: int, optional
            Version to select. If None, last version will be selected.
            If version given is larger than largest version available,
            the largest version will be selected.

        """
        configs = self.storage.fetch_experiments({"name": name})

        if not configs:
            return {}

        config = _fetch_config_version(configs, version)

        if len(configs) > 1 and version is None:
            log.info(
                "Many versions for experiment %s have been found. Using latest "
                "version %s.",
                name,
                config["version"],
            )

        log.debug("Config found in DB:\n%s", pprint.pformat(config))

        backward.populate_space(config, force_update=False)
        backward.update_max_broken(config)

        return config

    def _register_experiment(self, experiment: Experiment):
        """Register a new experiment in the database"""
        experiment.metadata["datetime"] = datetime.datetime.utcnow()
        config = experiment.configuration
        # This will raise DuplicateKeyError if a concurrent experiment with
        # identical (name, metadata.user) is written first in the database.

        self.storage.create_experiment(config)

        # XXX: Reminder for future DB implementations:
        # MongoDB, updates an inserted dict with _id, so should you :P
        experiment._id = config["_id"]  # pylint:disable=protected-access

        # Update refers in db if experiment is root
        if experiment.refers.get("parent_id") is None:
            log.debug("update refers (name: %s)", experiment.name)
            experiment.refers["root_id"] = experiment.id
            self.storage.update_experiment(
                experiment, refers=experiment.configuration["refers"]
            )

    def _update_experiment(self, experiment: Experiment) -> None:
        """Update experiment configuration in database"""
        log.debug("Updating experiment (name: %s)", experiment.name)
        config = experiment.configuration

        # TODO: Remove since this should not occur anymore without metadata.user in the indices?
        # Writing the final config to an already existing experiment raises
        # a DuplicatKeyError because of the embedding id `metadata.user`.
        # To avoid this `final_config["name"]` is popped out before
        # `db.write()`, thus seamingly breaking  the compound index
        # `(name, metadata.user)`
        config.pop("name")

        self.storage.update_experiment(experiment, **config)

        log.debug("Experiment configuration successfully updated in DB.")

    def _attempt_branching(self, conflicts, experiment, version, branching):
        if len(conflicts.get()) > 1:
            log.debug("Experiment must branch because of conflicts")
        else:
            assert branching.get("branch_to")
            log.debug("Experiment branching forced with ``branch_to``")

        branched_experiment = self._branch_experiment(
            experiment, conflicts, version, branching
        )
        log.debug("Now attempting registration of branched experiment in DB.")
        try:
            self._register_experiment(branched_experiment)
            log.debug("Branched experiment successfully registered in DB.")
        except DuplicateKeyError as e:
            log.debug(
                "Experiment registration failed. This is likely due to a race condition "
                "during branching. Now rolling back and re-attempting building "
                "the branched experiment."
            )
            raise RaceCondition(
                "There was a race condition during branching. This error can "
                "also occur if you try branching from a specific version that already "
                "has a child experiment with the same name. Change the name of the new "
                "experiment and use `branch-from` to specify the parent experiment."
            ) from e

        return branched_experiment

    def consolidate_config(self, name: str, version: int | None, config: dict):
        """Merge together given configuration with db configuration matching
        for experiment (``name``, ``version``)
        """
        db_config = self.fetch_config_from_db(name, version)

        # Do not merge spaces, the new definition overrides it.
        if "space" in config:
            db_config.pop("space", None)

        log.debug("Merging user and db configs:")
        log.debug("    config from user:\n%s", pprint.pformat(config))
        log.debug("    config from DB:\n%s", pprint.pformat(db_config))

        new_config = config
        config = resolve_config.merge_configs(db_config, config)

        config.setdefault("metadata", {})
        resolve_config.update_metadata(config["metadata"])

        merge_algorithm_config(config, new_config)
        # TODO: Remove for v0.4
        merge_producer_config(config, new_config)

        config.setdefault("name", name)
        config.setdefault("version", version)

        log.debug("    Merged config:\n%s", pprint.pformat(config))

        return config

    def _get_branching_status_string(self, conflicts, branching_arguments):
        experiment_brancher = ExperimentBranchBuilder(
            conflicts, enabled=False, storage=self.storage, **branching_arguments
        )
        branching_prompt = BranchingPrompt(experiment_brancher)
        return branching_prompt.get_status()

    def _branch_experiment(self, experiment, conflicts, version, branching_arguments):
        """Create a new branch experiment with adapters for the given conflicts"""
        experiment_brancher = ExperimentBranchBuilder(
            conflicts, storage=self.storage, **branching_arguments
        )

        needs_manual_resolution = (
            not experiment_brancher.is_resolved or experiment_brancher.manual_resolution
        )

        if not experiment_brancher.is_resolved:
            name_conflict = conflicts.get([ExperimentNameConflict])[0]
            if not name_conflict.is_resolved and not version:
                log.debug(
                    "A race condition likely occurred during conflicts resolutions. "
                    "Now rolling back and attempting re-building the branched experiment."
                )
                raise RaceCondition(
                    "There was likely a race condition during version increment."
                )

        if needs_manual_resolution:
            log.debug("Some conflicts cannot be solved automatically.")

            # TODO: This should only be possible when using cmdline API
            branching_prompt = BranchingPrompt(experiment_brancher)

            if not sys.__stdin__.isatty():
                log.debug(
                    "No interactive prompt available to manually resolve conflicts."
                )
                raise BranchingEvent(branching_prompt.get_status())

            branching_prompt.cmdloop()

            if branching_prompt.abort or not experiment_brancher.is_resolved:
                sys.exit()

        log.debug("Creating new branched configuration")
        config = experiment_brancher.conflicting_config
        config["refers"][
            "adapter"
        ] = experiment_brancher.create_adapters().configuration
        config["refers"]["parent_id"] = experiment.id

        config.pop("_id")

        return self.create_experiment(mode="x", **config)

    # too-many-locals disabled for the deprecation of algorithms.
    # We will be able to able once algorithms is removed
    # pylint: disable=too-many-arguments,too-many-locals
    def create_experiment(
        self,
        name: str,
        version: int,
        mode: Mode,
        space: Space | dict[str, str],
        algorithm: str | dict | None = None,
        algorithms: str | dict | None = None,
        max_trials: int | None = None,
        max_broken: int | None = None,
        working_dir: str | None = None,
        metadata: dict | None = None,
        refers: dict | None = None,
        producer: dict | None = None,
        knowledge_base: KnowledgeBase | dict | None = None,
        user: str | None = None,
        _id: int | str | None = None,
        **kwargs,
    ) -> Experiment:
        """Instantiate the experiment and its attribute objects

        All unspecified arguments will be replaced by system's defaults (orion.core.config.*).

        Parameters
        ----------
        name: str
            Name of the experiment.
        version: int
            Version of the experiment.
        mode: str
            The access rights of the experiment on the database.
            'r': read access only
            'w': can read and write to database
            'x': can read and write to database, algo is instantiated and can execute optimization
        space: dict or Space object
            Optimization space of the algorithm. If dict, should have the form
            `dict(name='<prior>(args)')`.
        algorithm: str or dict, optional
            Algorithm used for optimization.
        strategy: str or dict, optional
            Parallel strategy to use to parallelize the algorithm.
        max_trials: int, optional
            Maximum number or trials before the experiment is considered done.
        max_broken: int, optional
            Number of broken trials for the experiment to be considered broken.
        storage: dict, optional
            Configuration of the storage backend.
        knowledge_base: KnowledgeBase | dict, optional
            Knowledge base instance, or configuration of the knowledge base. Will be used to
            warm-start the HPO algorithm, if possible.

        """
        T = TypeVar("T")
        V = TypeVar("V")

        def _default(v: T | None, default: V) -> T | V:
            return v if v is not None else default

        space = _instantiate_space(space)
        max_trials = _default(max_trials, orion.core.config.experiment.max_trials)
        if isinstance(knowledge_base, dict):
            knowledge_base = _instantiate_knowledge_base(knowledge_base)

        instantiated_algorithm = _instantiate_algo(
            space=space,
            max_trials=max_trials,
            config=algorithm,
            ignore_unavailable=mode != "x",
            knowledge_base=knowledge_base,
        )
        if algorithms is not None and algorithm is None:
            log.warning(
                "algorithms is deprecated and will be removed in v0.4.0. Use algorithm instead."
            )
            instantiated_algorithm = _instantiate_algo(
                space=space,
                max_trials=max_trials,
                config=algorithms,
                ignore_unavailable=mode != "x",
                knowledge_base=knowledge_base,
            )

        max_broken = _default(max_broken, orion.core.config.experiment.max_broken)
        working_dir = _default(working_dir, orion.core.config.experiment.working_dir)
        metadata = _default(metadata, {"user": _default(user, getpass.getuser())})
        refers = _default(refers, dict(parent_id=None, root_id=None, adapter=[]))
        refers["adapter"] = _instantiate_adapters(refers.get("adapter", []))  # type: ignore

        _instantiate_strategy((producer or {}).get("strategy"))

        experiment = Experiment(
            storage=self.storage,
            name=name,
            version=version,
            mode=mode,
            space=space,
            _id=_id,
            max_trials=max_trials,
            algorithm=instantiated_algorithm,
            max_broken=max_broken,
            working_dir=working_dir,
            metadata=metadata,
            refers=refers,
            knowledge_base=knowledge_base,
        )
        if kwargs:
            # TODO: https://github.com/Epistimio/orion/issues/972
            log.debug(
                "create_experiment received some extra unused arguments: %s", kwargs
            )

        return experiment
