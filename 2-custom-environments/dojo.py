import numpy as np
import torch

arr = np.random.rand(10)
print("Numpy array:", arr)

tensor = torch.tensor(arr)
print("Torch tensor:", tensor)