import numpy as np

import torch
from torch import nn

class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()

        # Layer parameters
        self.input_dim = input_dim
        
        # Fully connected layer
        self.lin = nn.Linear(self.input_dim, 1)


    def forward(self, x):
        return self.lin(x)