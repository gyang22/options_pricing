import torch
import torch.nn as nn
import torch.nn.functional as F


class PricingModel(nn.Module):

    def __init__(self, input_dim: int):
        super(PricingModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, 64)
        self.h1 = nn.Linear(64, 64)
        self.h2 = nn.Linear(64, 64)
        self.h3 = nn.Linear(64, 64)
        self.h4 = nn.Linear(64, 64)
        self.h5 = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, 1)

        self.dropout = nn.Dropout(p=0.1)

    
    def forward(self, x):
        x = F.leaky_relu(self.input_layer(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.h1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.h2(x))
        x = self.dropout(x)
        # x = F.leaky_relu(self.h3(x))
        # x = self.dropout(x)
        # x = F.leaky_relu(self.h4(x))
        # x = self.dropout(x)
        # x = F.leaky_relu(self.h5(x))
        # x = self.dropout(x)
        return F.relu(self.output_layer(x))
        