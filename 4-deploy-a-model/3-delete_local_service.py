from azureml.core import Workspace
from azureml.core.webservice import LocalWebservice

SERVICE_NAME = "cifar10net-endpoint"

ws = Workspace.from_config()
service = LocalWebservice(ws, name=SERVICE_NAME)
if service:
    service.delete()
    print("Deleted existing service")
