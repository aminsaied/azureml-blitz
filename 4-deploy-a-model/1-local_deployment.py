import argparse
import os

from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import Webservice
from azureml.core.webservice import LocalWebservice
from azureml.exceptions import WebserviceException

SERVICE_NAME = "cifar10net-endpoint"
MODEL_NAME = "cifar10net"
LOCAL_MODEL_DIR = "../outputs"


ws = Workspace.from_config()

deployment_config = None

model_path = os.path.join(LOCAL_MODEL_DIR, "net.pt")
try:
    print(f"Getting registered model {MODEL_NAME}")
    registered_model = None
except WebserviceException:
    print("Register model...")
    registered_model = None

env = None

inference_config = None

try:
    service = None
    if service:
        service.delete()
        print("Deleted existing service")
except WebserviceException as e:
    pass

print("Deploying model to service...")
service = None
service.wait_for_deployment(True)

# output scoring url
print(service.scoring_uri)
