"""Submit our first job to Azure ML!"""

from azureml.core import Workspace, Experiment, ScriptRunConfig

# get workspace
ws = None

# get compute target
target = None

# set up script run configuration
config = None

# get/create experiment
exp = None

# submit script to AML
run = exp.submit(config)
run.set_tags({"part": "0"})
print(run.get_portal_url()) # link to ml.azure.com
run.wait_for_completion(show_output=True)
