from string import ascii_lowercase


def validate_hyperparameters(response):
    """Check to make sure that server HP values are in range"""
    hps = response.get_json()["hyperparameters"]
    assert 5e-6 <= hps["learning_rate"] <= 5e-4
    assert hps["batch_size"] in [32, 64, 128]


def test_app(
    client,
    objective,
    max_trials,
    output_dir,
    project_name,
    max_parallel_workers,
):
    # make sure that we can get the hyperparameter
    # names from the server
    url = "http://localhost:5000"
    response = client.get(f"{url}/hyperparameters")
    hp_names = response.get_json()["hyperparameters"]
    assert hp_names == ["learning_rate", "batch_size"]

    # create one too many workers to make sure
    # that the last one gets rejected
    worker_ids = ascii_lowercase[: max_parallel_workers + 1]
    for worker_id in worker_ids:
        response = client.get(f"{url}/start/{worker_id}")

        if worker_id == worker_ids[-1]:
            assert response.status == "400 BAD REQUEST"
        else:
            # make sure the response has all the appropriate info
            validate_hyperparameters(response)
            trial_id = response.get_json()["id"]

    # verify that the "ongoing" target returns
    # the correct trial id for the last worker
    response = client.get(f"{url}/ongoing/{worker_ids[-2]}")
    assert response.get_json()["id"] == trial_id

    # now make sure that subsequent calls to "end"
    # after the last trial will return a blank trial id
    for i in range(max_trials - max_parallel_workers + 1):
        response = client.get(
            f"{url}/end/{trial_id}",
            query_string={"val_loss": 10 ** -i, "worker_id": worker_ids[-2]},
        )
        trial_id = response.get_json()["id"]
        if i == (max_trials - max_parallel_workers):
            assert trial_id == ""
        else:
            validate_hyperparameters(response)
            assert trial_id != ""
