import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNorm(nn.Module):
    def __init__(self, num_features, num_classes, beta=10):
        super(FCNorm, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.beta = beta

    def forward(self, x):
        out = self.beta * F.linear(F.normalize(x), F.normalize(self.weight))
        return out

class NonLinearNeck(nn.Module):
    def __init__(self, num_features=2048, num_out=128, hidden_dim=512):
        super(NonLinearNeck, self).__init__()
        self.layer_1 = nn.Linear(num_features, hidden_dim)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        return self.layer_2(self.relu(self.layer_1(x)))
    
class LinearNeck(nn.Module):
    def __init__(self, num_features=2048, num_out=128):
        super(LinearNeck, self).__init__()
        self.layer_1 = nn.Linear(num_features, num_out)

    def forward(self, x):
        return self.layer_1(x)
