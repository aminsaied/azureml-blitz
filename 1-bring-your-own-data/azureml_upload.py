from azureml.core import Workspace

ws = Workspace.from_config()
ds = ws.get_default_datastore()

ds.upload_files(
    ["./wisdom.txt"],
    target_path="azureml-blitz",
    overwrite=True,
)