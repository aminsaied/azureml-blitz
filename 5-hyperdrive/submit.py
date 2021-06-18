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
        "--learning_rate": None,
        "--momentum": None,
    }
    hyperparameter_sampling = None

    hyperdrive_config = None

    run = Experiment(ws, "azureml-blitz").submit(hyperdrive_config)
    run.set_tags({"part": "5"})
    print(run.get_portal_url()) # link to ml.azure.com
