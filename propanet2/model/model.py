import argparse

import torch
import torch.nn as nn
import torch_geometric.nn as geometric_nn

class DEGInfluence(nn.Module):
    def __init__(self):
        super(DEGInfluence, self).__init__()

class GRNPropagation(nn.Module):
    def __init__(self, num_layers:int, alpha:float):
        super(GRNPropagation, self).__init__()
        
        self.propagation = geometric_nn.models.LabelPropagation(num_layers, alpha)
        self.correct_and_smooth = geometric_nn.models.CorrectAndSmooth()

    def forward(self, y:torch.Tensor, edge_index:torch.Tensor, edge_weight:torch.Tensor):
        y_hat = self.propagation(y, edge_index, edge_weight)

        return y_hat

if __name__ == "__main__":
    pass