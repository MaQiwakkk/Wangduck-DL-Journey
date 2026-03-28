import torch
from torch import nn
from torch.nn import L1Loss, MSELoss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss_L1 = L1Loss(reduction="sum")  # reduction="mean"
result_L1 = loss_L1(inputs, targets)

loss_mse = MSELoss()
result_mse = loss_mse(inputs, targets)

print(result_L1)
print(result_mse)
