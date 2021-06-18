from azureml.core import (
    Workspace,
    Experiment,
    Environment,
    ScriptRunConfig,
)

# get workspace
ws = Workspace.from_config()
target = ws.compute_targets['cpucluster']

# create environment
env = Environment.from_pip_requirements('blitz-env', 'requirements.txt')

# set up script run configuration
config = ScriptRunConfig(
    source_directory='.',
    script='dojo.py',
    compute_target=target,
    environment=env,
)

# submit script to AML
exp = Experiment(ws, "azureml-blitz")
run = exp.submit(config)
run.set_tags({"part": "2"})
print(run.get_portal_url()) # link to ml.azure.com
run.wait_for_completion(show_output=True, raise_on_error=True)