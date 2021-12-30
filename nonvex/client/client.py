import importlib
import inspect
import re
import shutil
import sys
from dataclasses import dataclass
from functools import partial
from secrets import token_hex
from typing import Callable, Dict, List, Optional

import requests
from hermes.typeo import typeo


@dataclass
class NonvexClient:
    url: str
    worker_id: Optional[str] = None

    def __post_init__(self):
        if self.worker_id is None:
            self.worker_id = token_hex(15)

    def get_hyperparameters(self):
        response = requests.get(f"{self.url}/hyperparameters")
        response.raise_for_status()
        return response.json()["hyperparameters"]

    def _read_response(self, response):
        response = response.json()
        if response["id"] == "":
            return None, None
        return response["hyperparameters"], response["id"]

    def start_worker(self):
        response = requests.get(f"{self.url}/start/{self.worker_id}")
        try:
            response.raise_for_status()
        except requests.HTTPError:
            if response.code == 400:
                raise RuntimeError(
                    "Too many parallel workers in progress for "
                    "hyperparameter server at URL {}".format(self.url)
                )
            raise

        return self._read_response(response)

    def end_trial(self, trial_id, result):
        params = {"worker_id": self.worker_id}
        params.update(result)

        response = requests.get(f"{self.url}/end/{trial_id}", params=params)
        response.raise_for_status()
        return self._read_response(response)


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
    signature = inspect.signature(fn)
    parameters = []
    for param_name, param in signature.parameters.items():
        if param_name not in hyperparameters:
            parameters.append(param)

    def spoof_fn(**kwargs):
        return kwargs

    spoof_fn.__signature__ = inspect.Signature(parameters=parameters)
    spoof_fn.__doc__ = fn.__doc__

    sys.argv = [None] + args
    return typeo(spoof_fn)()


def main(
    executable: str,
    url: str = "localhost:5000",
    worker_id: Optional[str] = None,
) -> List[Dict[str, float]]:
    client = NonvexClient(url, worker_id)
    hyperparameters = client.get_hyperparameters()

    fn = get_train_fn(executable)
    args = {}
    kwargs = read_fn_kwargs(fn, args, hyperparameters)

    train_fn = partial(fn, **kwargs)
    hyperparameters, trial_id = client.start_worker()
    results = []
    while trial_id is not None:
        result = train_fn(**hyperparameters)
        results.append(result)
        hyperparameters, trial_id = client.end_trial(trial_id, result)
    return results


if __name__ == "__main__":
    main()
