import importlib
import inspect
import os
import re
import shutil
import sys
from typing import Callable, Dict, List, Optional

from hermes.typeo import typeo

from nonvex.search.client import NonvexClient


def get_train_fn(executable: str) -> Callable:
    try:
        library, fn = executable.split(":")
    except ValueError:
        executable_path = shutil.which(executable)
        import_re = re.compile(
            "(?m)^from (?P<lib>[a-zA-Z0-9_.]+) import (?P<fn>[a-zA-Z0-9_]+)$"
        )
        with open(executable_path, "r") as f:
            match = import_re.search(f.read())
            if match is None:
                raise ValueError(
                    "Could not find library to import in "
                    "executable at path {}".format(executable_path)
                )
            library = match.group("lib")
            fn = match.group("fn")

    module = importlib.import_module(library)
    return getattr(module, fn)


def read_fn_kwargs(
    fn: Callable, args: List[str], hyperparameters: List[str]
) -> Dict:
    """
    Use `fn`'s signature to parse out any command line arguments
    using `typeo`. Any arguments being search over as hyper-
    parameters will be dropped.
    """

    # rather than ignore hyperparameters altogether,
    # we want to keep them in the signature so that
    # if they're contained in e.g. a typeo config,
    # the parser won't complain about extra arguments.
    # Instead, we'll just given them default values of
    # `None` and place them at the end of the parameter
    # list, then drop them after parsing.
    signature = inspect.signature(fn)
    parameters = []
    hp_parameters = []
    for param_name, param in signature.parameters.items():
        if param_name in hyperparameters:
            # if this is a hyperparameter argument, replace
            # it with a dummy parameter that default to None
            # and record it in our `hp_parameters` list that
            # we'll tack on to the end of `parameters`
            param = inspect.Parameter(
                name=param.name,
                annotation=param.annotation,
                default=None,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            hp_parameters.append(param)
        else:
            parameters.append(param)

    # put hyperparameters in the back since they
    # have defaults now
    parameters = parameters + hp_parameters

    # create a dummy function that just returns
    # the pass command line arguments as a dictionary,
    # and give it the same signature as `fn` so that
    # typeo knows the names and types of the arguments
    # to process
    def spoof_fn(**kwargs):
        return kwargs

    spoof_fn.__signature__ = inspect.Signature(parameters=parameters)
    spoof_fn.__doc__ = fn.__doc__
    spoof_fn.__name__ = fn.__name__

    # parse the command line arguments with typeo and then
    # pop out any hyperparameters that may have been
    # in there because e.g. they were in a typeo config
    sys.argv = [None] + args
    kwargs = typeo(spoof_fn)()
    for hp in hyperparameters:
        try:
            kwargs.pop(hp)
        except KeyError:
            continue

    # return these kwargs as a dict
    return kwargs


def run_search(
    executable: str,
    url: str = "http://localhost:5000",
    worker_id: Optional[str] = None,
    max_fails: int = 5,
    args: Optional[List[str]] = None,
) -> List[Dict[str, float]]:
    """Run a hyperparameter search over a training function

    Args:
        executable:
            The training executable or function to search over
        url:
            The URL of the hyperparameter server
        worker_id:
            A unique ID to assign to this worker. If left as `None`,
            a random hex value will be assigned
        args:
            Any command line arguments to pass to `executable`
    """

    client = NonvexClient(url, worker_id)
    os.environ["NV_WORKER_ID"] = client.worker_id
    hyperparameters = client.get_hyperparameters()

    fn = get_train_fn(executable)
    hyperparameters, trial_id = client.start_worker()
    results = []
    while trial_id is not None:
        # do command line argument parsing inside of the
        # loop in case we reference any nonvex environment
        # variables in the arguments
        os.environ["NV_TRIAL_ID"] = trial_id
        kwargs = read_fn_kwargs(fn, args or [], hyperparameters)
        kwargs.update(hyperparameters)

        try:
            result = fn(**kwargs)
        except Exception:
            hyperparameters, trial_id = client.cancel_trial()
            if trial_id is None:
                raise
            continue

        results.append(result)
        hyperparameters, trial_id = client.end_trial(trial_id, result)
    return results
