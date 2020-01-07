import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp


class PermutateGraph(nn.Module):
    def __init__(self):
        super(PermutateGraph, self).__init__()

    def forward(self, G):
        features = G.inputs
        # features = features.todense()

        n = features.shape[0]

        idx = torch.randperm(n)
        shuffled_features = features[idx, :]
        # G.inputs = shuffled_features

        return shuffled_features