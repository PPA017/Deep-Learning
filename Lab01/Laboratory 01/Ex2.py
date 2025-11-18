import torch
from torch import nn
import matplotlib.pyplot as plt

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    
    def forward(self, x : torch.Tensor)->torch.Tensor:
        return self.weights * x + self.bias
    
torch.manual_seed(42)

model_1 = LinearRegressionModel()

print(f"The model's state dict(): {model_1.state_dict()}")
 
        