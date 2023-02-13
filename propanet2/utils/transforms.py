import torch

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

class DropLowWeightEdges(BaseTransform):
    def __init__(self, cutoff_weight:float):
        super(DropLowWeightEdges, self).__init__()
        self.cutoff_weights = cutoff_weight

    def __call__(self, data:Data)->Data:
        for store in data.edge_stores:
            edge_index = store.edge_index
            edge_attr = store.edge_attr

            mask = edge_attr >= self.cutoff_weights
            indices = torch.nonzero(torch.squeeze(mask, dim = 1)).squeeze()

            store.edge_index = torch.index_select(edge_index, 1, indices)

        return data

if __name__ == "__main__":
    num_nodes = 100
    num_edges = 200

    x = torch.randn((num_nodes, 12))
    edge_index = torch.randint(num_nodes, (2, num_edges))
    edge_attr = torch.randn((num_edges, 1))

    data = Data(x, edge_index=edge_index, edge_attr=edge_attr)
    print(data.num_edges)

    droplayer = DropLowWeightEdges(0.5)
    data = droplayer(data)
    print(data.num_edges)