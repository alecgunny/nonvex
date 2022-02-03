import os
import shutil

import pytest

from nonvex.app import create_app

scope = "function"


@pytest.fixture(scope=scope, autouse=True)
def config():
    content = """
import keras_tuner as kt

hyperparameters = kt.HyperParameters()
hyperparameters.Float("learning_rate", 5e-6, 5e-4, sampling="log")
hyperparameters.Choice("batch_size", [32, 64, 128])
"""

    with open("nonvex-hp.py", "w") as f:
        f.write(content)
    yield
    os.remove("nonvex-hp.py")


@pytest.fixture(scope=scope)
def objective():
    return "val_loss"


@pytest.fixture(scope=scope)
def max_trials():
    return 10


@pytest.fixture(scope="session")
def output_dir():
    yield "test"
    shutil.rmtree("test")


@pytest.fixture(scope=scope)
def project_name():
    return "app-test"


@pytest.fixture(scope=scope)
def max_parallel_workers():
    return 4


@pytest.fixture(scope=scope, autouse=True)
def app(
    config,
    objective,
    max_trials,
    output_dir,
    project_name,
    max_parallel_workers,
):
    app = create_app(
        objective=objective,
        max_trials=max_trials,
        output_dir=output_dir,
        project_name=project_name,
        max_parallel_workers=max_parallel_workers,
    )
    return app
