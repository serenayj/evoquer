
"""
src: https://github.com/eriklindernoren/Action-Recognition/blob/master/models.py
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable


class AcClassifier(nn.Module):
    def __init__(self, in_features, num_classes, latent_dim):
        super(AcClassifier, self).__init__()
        self.final = nn.Sequential(
            nn.Linear(in_features, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=0.01),
            nn.Linear(latent_dim, num_classes),
            #nn.Sigmoid(dim=-1),
        )

    def forward(self, x):
        x = self.final(x)
        x = torch.sigmoid(x) 
        return x


class ObjClassifier(nn.Module):
    def __init__(self, in_features, num_classes, latent_dim):
        super(ObjClassifier, self).__init__()
        self.final = nn.Sequential(
            nn.Linear(in_features, latent_dim),
            nn.BatchNorm1d(latent_dim, momentum=0.01),
            nn.Linear(latent_dim, num_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.final(x)
        return x