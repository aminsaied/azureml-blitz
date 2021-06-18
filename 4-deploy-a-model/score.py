import json
import os
import pickle

import numpy as np
import torch

from model import Net

def init():
    global net
    path = os.path.join(os.environ.get("AZUREML_MODEL_DIR", "../outputs"), "net.pt")
    net = Net()
    net.load_state_dict(torch.load(path))
    net.eval()

def run(raw_data):
    data = json.loads(raw_data)
    inputs = data["inputs"]
    inputs = torch.tensor(inputs)

    outputs = net(inputs)

    _, predictions = torch.max(outputs, 1)
    return predictions.tolist()