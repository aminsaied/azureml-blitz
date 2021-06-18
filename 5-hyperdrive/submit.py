import math
from azureml.core import (
    Experiment,
    Workspace,
    ScriptRunConfig,
)
from azureml.train.hyperdrive import (
    RandomParameterSampling,
    BanditPolicy,
    HyperDriveConfig,
)
from azureml.train import hyperdrive

if __name__ == "__main__":
    ws = Workspace.from_config()

    target = ws.compute_targets["gpu-K80-2"]

    env = ws.environments['AzureML-PyTorch-1.6-GPU']

    config = ScriptRunConfig(
        source_directory=".",
        script="train.py",
        compute_target=target,
        environment=env,
        arguments=["--output_dir", "outputs", "--num_epochs", 5],
    )

    convert = lambda x: float(math.log(x))
    search_space = {
        "--learning_rate": hyperdrive.loguniform(convert(1e-10), convert(1e-3)),
        "--momentum": hyperdrive.uniform(0.5, 1),
    }
    hyperparameter_sampling = RandomParameterSampling(search_space)

    hyperdrive_config = HyperDriveConfig(
        run_config=config,
        hyperparameter_sampling=hyperparameter_sampling,
        primary_metric_name="loss",
        primary_metric_goal=hyperdrive.PrimaryMetricGoal.MAXIMIZE,
        max_total_runs=10,
        max_concurrent_runs=10,
    )

    run = Experiment(ws, "azureml-blitz").submit(hyperdrive_config)
    run.set_tags({"part": "5"})
    print(run.get_portal_url()) # link to ml.azure.com
