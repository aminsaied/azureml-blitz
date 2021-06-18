"""Submit our first job to Azure ML!"""

from azureml.core import Workspace, Experiment, ScriptRunConfig

# get workspace
ws = Workspace.from_config()

# get compute target
target = ws.compute_targets['cpucluster']

# set up script run configuration
config = ScriptRunConfig(
    source_directory='.',   # source code to upload to the remote compute
    script='hello.py',      # the entry point
    compute_target=target,  # azureml compute
)

# get/create experiment
exp = Experiment(ws, 'azureml-blitz')

# submit script to AML
run = exp.submit(config)
run.set_tags({"part": "0"})
print(run.get_portal_url()) # link to ml.azure.com
run.wait_for_completion(show_output=True)
