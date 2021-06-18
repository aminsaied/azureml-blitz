from azureml.core import (
    Dataset,
    Workspace,
    Experiment,
    ScriptRunConfig,
)

# connect to azureml assets
ws = Workspace.from_config()
target = ws.compute_targets["cpucluster"]

# create dataset
datastore = None
dataset = None

# pass dataset as argument
arguments = None

# set up script run configuration
config = ScriptRunConfig(
    source_directory='.',
    script='read_data.py',
    compute_target=target,
    arguments=arguments,
)

# submit script to AML
exp = Experiment(ws, "azureml-blitz")
run = exp.submit(config)
run.set_tags({"part": "1"})
print(run.get_portal_url()) # link to ml.azure.com
run.wait_for_completion(show_output=True, raise_on_error=True)