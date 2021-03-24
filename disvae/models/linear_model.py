import numpy as np

import torch
from torch import nn

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        
        # Fully connected layer
        self.lin = nn.Linear(input_dim, output_dim)
        self.log_softmax = torch.nn.LogSoftmax()


    def forward(self, x):
        return self.log_softmax(self.lin(x))