import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

import pdb

class CrossEntropy(nn.Module):
    def __init__(self, para_dict=None):
        super(CrossEntropy, self).__init__()

    def forward(self, output, target, reduction='mean'):
        output = output
        loss = F.cross_entropy(output, target, reduction=reduction)
        return loss
