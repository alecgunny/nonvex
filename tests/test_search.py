import os
from unittest.mock import Mock, patch

import pytest
import toml

from nonvex import search


@pytest.fixture
def worker_id():
    worker_id = "paranoid-android"
    yield worker_id

    if os.path.exists(worker_id + ".log"):
        os.remove(worker_id + ".log")


@pytest.fixture(autouse=True)
def train_script():
    content = """
def main(
    learning_rate: float, batch_size: int, hidden_dim: int
):
    return {"val_loss": learning_rate}
"""

    with open("train.py", "w") as f:
        f.write(content)

    yield
    os.remove("train.py")


def test_search(client, max_trials):
    def get_patch(url, params=None):
        response = client.get(url, query_string=params)
        mock = Mock()
        mock.raise_for_status = lambda: None
        mock.json = lambda: response.get_json()
        return mock

    with patch("requests.get", get_patch):
        # make sure we can parse the function from the name
        fn = search.search.get_train_fn("train:main")
        assert fn(1e-3, 32, 128)["val_loss"] == 1e-3

        # make sure that our request to the
        # server works as expected
        nv_client = search.client.NonvexClient("http://localhost:5000")
        hps = nv_client.get_hyperparameters()
        assert hps == ["learning_rate", "batch_size"]

        # make sure that the kwarg parser comes
        # up with the right arguments
        args = ["--hidden-dim", "128"]
        kwargs = search.search.read_fn_kwargs(fn, args, hps)
        assert len(kwargs) == 1
        assert kwargs["hidden_dim"] == 128

        # finally run a search and make sure all
        # trials get run, and that the learning
        # rate (stored as "val_loss") is within
        # range for all runs
        results = search.search.run_search("train:main", args=args)
        assert len(results) == max_trials

        for i in results:
            assert 5e-6 < i["val_loss"] < 5e-4


@pytest.fixture
def typeo_config():
    config = {
        "typeo": {
            "hidden_dim": 256,
            "learning_rate": 10,
            "log_file": "${NV_WORKER_ID}.log",
        }
    }
    with open("config.toml", "w") as f:
        toml.dump(config, f)
    yield "config.toml"

    os.remove("config.toml")


@pytest.fixture(autouse=True)
def train_script_with_log():
    content = """
def main(
    learning_rate: float, batch_size: int, hidden_dim: int, log_file: str
):
    with open(log_file, "w") as f:
        f.write("Please can you stop the noise")
    return {"val_loss": learning_rate}
"""

    with open("train_with_log.py", "w") as f:
        f.write(content)

    yield
    os.remove("train_with_log.py")


def test_search_with_typeo_config(client, worker_id, typeo_config):
    def get_patch(url, params=None):
        response = client.get(url, query_string=params)
        mock = Mock()
        mock.raise_for_status = lambda: None
        mock.json = lambda: response.get_json()
        return mock

    with patch("requests.get", get_patch):
        # test to make sure we can run a search with a
        # typeo config for the searched function, including
        # using NV_* environment variables
        results = search.search.run_search(
            "train_with_log:main",
            worker_id=worker_id,
            args=["--typeo", "config.toml"],
        )

        assert len(results) > 0
        for i in results:
            assert 5e-6 < i["val_loss"] < 5e-4

        assert os.path.exists(worker_id + ".log")
        with open(worker_id + ".log", "r") as f:
            assert f.read() == "Please can you stop the noise"
