from inspect import ArgInfo
from azureml.core import (
    Experiment,
    Workspace,
    ScriptRunConfig,
)

if __name__ == "__main__":
    ws = Workspace.from_config()

    target = ws.compute_targets["gpu-nc6-ssh"]

    env = ws.environments['AzureML-PyTorch-1.6-GPU']

    config = ScriptRunConfig(
        source_directory=".",
        script="train.py",
        compute_target=target,
        environment=env,
        arguments=["--output_dir", "outputs"],
    )

    run = Experiment(ws, "test-intune-demo").submit(config)
    run.set_tags({"part": "3"})
    print(run.get_portal_url())
