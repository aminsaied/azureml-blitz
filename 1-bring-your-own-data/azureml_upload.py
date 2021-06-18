from azureml.core import Workspace

ws = Workspace.from_config()

# connect to default datastore
# upload data