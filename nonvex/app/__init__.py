from dataclasses import dataclass

import keras_tuner as kt
from flask import Flask, make_response, request


def _load_hyperparameters():
    locals_dict = {}
    with open("nonvex.py", "r") as f:
        exec(f.read(), {}, locals_dict)
    try:
        return locals_dict["hyperparameters"]
    except KeyError:
        raise ValueError("'nonvex.py' has no variable 'hyperparameters'")


@dataclass
class Searcher:
    objective: str
    max_trials: int
    output_dir: str
    project_name: str
    max_parallel_workers: int = 1
    max_fails_per_worker: int = 5

    def __post_init__(self):
        hyperparameters = _load_hyperparameters()
        self.oracle = kt.oracles.RandomSearch(
            objective=self.objective,
            max_trials=self.max_trials,
            hyperparameters=hyperparameters,
        )
        self.oracle._set_project_dir(
            self.output_dir, self.project_name, overwrite=True
        )
        self.failure_counts = {}

    def get_hyperparameters(self):
        hps = list(self.oracle.hyperparameters._hps.keys())
        return make_response({"hyperparameters": hps})

    def create_trial(self, worker_id):
        trial = self.oracle.create_trial(worker_id)

        # empty hyperparameters means that we've exceeded
        # the max number of trials, so send back blank
        # data to indicate that a client should stop
        if not trial.hyperparameters.values:
            data = {"id": "", "hyperparameters": {}}
        else:
            data = {
                "id": trial.trial_id,
                "hyperparameters": trial.hyperparameters.values,
            }
        return make_response(data)

    def begin_worker(self, worker_id):
        """Create an initial trial for a new worker"""

        # if we have too many parallel workers already, then
        # reject this worker and send back an HTTP error
        if len(self.oracle.ongoing_trials) >= self.max_parallel_workers:
            return "Too many workers", 400
        self.failure_counts[worker_id] = 0

        # create an initial trial for this worker
        return self.create_trial(worker_id)

    def get_trial_id(self, worker_id):
        try:
            trial_id = self.oracle.ongoing_trials[worker_id].trial_id
        except KeyError:
            trial_id = ""
        return make_response({"id": trial_id})

    def cancel_trial(self, worker_id):
        trial = self.oracle.ongoing_trials.pop(worker_id)
        self.oracle.trials.pop(trial.trial_id)

        self.failure_counts[worker_id] += 1
        if self.failure_counts[worker_id] >= self.max_fails_per_worker:
            return {"id": "", "hyperparameters": {}}
        return self.create_trial(worker_id)

    def end_trial(self, trial_id, result, worker_id):
        """End an existing trial and potentially start a new one"""
        self.oracle.update_trial(trial_id, {self.objective: result})

        trial = self.oracle.trials[trial_id]
        trial.status = kt.engine.trial.TrialStatus.COMPLETED
        self.oracle.end_trial(trial_id)

        return self.create_trial(worker_id)


def create_app(
    objective: str,
    max_trials: int,
    output_dir: str,
    project_name: str,
    max_parallel_workers: int,
    max_fails_per_worker: int = 5,
):
    """Start a Nonvex hyperparameter server

    Start a server which assigns hyperparameter values
    to client workers, indexed by unique trial ids.

    Args:
        objective:
            The name of the objective that workers will
            report back to the server
        max_trials:
            The naximum number of trials to run for the
            entire hyperparameter search
        output_dir:
            The output directory in which to create a
            project directory for this hyperparameter search
        project_name:
            The name to assign to this hyperparameter search
        max_parallel_workers:
            The maximum number of workers that can
            run trials simultaneously
    """

    app = Flask(__name__)
    searcher = Searcher(
        objective=objective,
        max_trials=max_trials,
        output_dir=output_dir,
        project_name=project_name,
        max_parallel_workers=max_parallel_workers,
        max_fails_per_worker=max_fails_per_worker,
    )

    def end_trial(trial_id):
        result = float(request.args.get(searcher.objective))
        worker_id = request.args.get("worker_id")
        return searcher.end_trial(trial_id, result, worker_id)

    app.route("/hyperparameters")(searcher.get_hyperparameters)
    app.route("/start/<worker_id>")(searcher.begin_worker)
    app.route("/ongoing/<worker_id>")(searcher.get_trial_id)
    app.route("/end/<trial_id>")(end_trial)
    app.route("/cancel/<worker_id>")(searcher.cancel_trial)

    return app
