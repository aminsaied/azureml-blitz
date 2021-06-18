import json
import requests

import torch
import torchvision


MODEL_DIR = "../outputs"

# download CIFAR 10 data
testset = torchvision.datasets.CIFAR10(
   root="../data",
   train=False,
   download=True,
   transform=torchvision.transforms.ToTensor(),
)
testloader = torch.utils.data.DataLoader(
   testset, batch_size=4, shuffle=True, num_workers=2
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(testloader)
images, labels = dataiter.next()

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

body = {"inputs": images.tolist()}
r = requests.post("http://localhost:8890/score", json=body)
predicted = r.json()
print(predicted)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
