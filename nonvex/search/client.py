from dataclasses import dataclass
from secrets import token_hex
from typing import Optional

import requests


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

    def cancel_trial(self):
        response = requests.get(f"{self.url}/cancel/{self.worker_id}")
        response.raise_for_status()
        return self._read_response(response)
