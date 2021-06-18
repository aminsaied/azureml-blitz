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

deployment_config = LocalWebservice.deploy_configuration(port=8890)

model_path = os.path.join(LOCAL_MODEL_DIR, "net.pt")
try:
    print(f"Getting registered model {MODEL_NAME}")
    registered_model = Model(ws, MODEL_NAME)
except WebserviceException:
    print("Register model...")
    registered_model = Model.register(
        workspace=ws,
        model_name=MODEL_NAME,
        model_path=model_path,
        model_framework="PyTorch",
        description="cifar10-net",
    )

env = ws.environments["AzureML-pytorch-1.7-ubuntu18.04-py37-cpu-inference"]

inference_config = InferenceConfig(
    source_directory=".",
    entry_script="score.py",
    environment=env,
)

try:
    service = Webservice(ws, name=SERVICE_NAME)
    if service:
        service.delete()
        print("Deleted existing service")
except WebserviceException as e:
    print()

print("Deploying model to service...")
service = Model.deploy(
    workspace=ws,
    name=SERVICE_NAME,
    models=[registered_model],
    inference_config=inference_config,
    deployment_config=deployment_config,
    overwrite=True,
    show_output=True,
)
service.wait_for_deployment(True)

# output scoring url
print(service.scoring_uri)
