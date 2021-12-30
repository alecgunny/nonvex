import os
from unittest.mock import Mock, patch

import pytest

from nonvex import client as client_lib


@pytest.fixture(autouse=True)
def train_script():
    content = """
def main(learning_rate: float, batch_size: int, hidden_dim: int):
    return {"val_loss": learning_rate}
"""

    with open("train.py", "w") as f:
        f.write(content)

    yield
    os.remove("train.py")


def test_client(client, max_trials):
    def get_patch(url, params=None):
        response = client.get(url, query_string=params)
        mock = Mock()
        mock.raise_for_status = lambda: None
        mock.json = lambda: response.get_json()
        return mock

    with patch("requests.get", get_patch):
        fn = client_lib.get_train_fn("train:main")
        assert fn(1e-3, 32, 128)["val_loss"] == 1e-3

        nonvex_client = client_lib.NonvexClient("http://localhost:5000")
        hps = nonvex_client.get_hyperparameters()
        assert hps == ["learning_rate", "batch_size"]

        args = ["--hidden-dim", "128"]
        kwargs = client_lib.read_fn_kwargs(fn, args, hps)
        assert len(kwargs) == 1
        assert kwargs["hidden_dim"] == 128

        results = client_lib.search("train:main", args=args)
        assert len(results) == max_trials

        for i in results:
            assert 5e-6 < i["val_loss"] < 5e-4
