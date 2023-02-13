import argparse
import sys
import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch_geometric.data import Data

from grn.build import get_adjmatrix
from utils.transforms import DropLowWeightEdges

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='python %(prog)s -nwk nwkFile -exp expFile -o out')
    parser.add_argument('-nwk', required=True, help='Network file')
    parser.add_argument('-exp', required=True, help='gene expression File')
    parser.add_argument('-ppi', required=True, help='protein-protein interaction network')
    parser.add_argument('-o', required=True, help='out File')
    args = parser.parse_args()

    network_edge = args.nwk
    network_ppi = args.ppi

    gene_exp = pd.read_csv(args.exp, sep = "\t", index_col=0)
    
    tgt_list = list(gene_exp.index)    
    # adj_matrix = create_adjmatrix(network_edge, tgt_list)

    # construct GRN
    x = torch.tensor(gene_exp.to_numpy())
    node_index = dict()

    for i, node in enumerate(tgt_list):
        node_index[node] = i

    grn_edge_list = list()
    grn_edge_attr_list = list()

    with open(network_edge) as f:
        for line in f.readlines():
            start, end, weight  = line.strip().split()

            grn_edge_list.append([node_index[start], node_index[end]])
            grn_edge_attr_list.append(float(weight))

    ppi_edge_list = list()
    ppi_edge_attr_list = list()

    with open(network_ppi) as f:
        for line in f.readlines():
            start, end, weight = line.strip().split()

            ppi_edge_list.append([node_index[start], node_index[end]])
            ppi_edge_attr_list.append(float(weight))
            
    grn_edge_index = torch.tensor(np.array(grn_edge_list).transpose())
    grn_edge_attr = torch.tensor(np.expand_dims(np.array(grn_edge_attr_list), axis = 1))

    ppi_edge_index = torch.tensor(np.array(ppi_edge_list).transpose())
    ppi_edge_attr = torch.tensor(np.expand_dims(np.array(ppi_edge_attr_list), axis = 1))    

    grn = Data(x = x, edge_index=grn_edge_index, edge_attr=grn_edge_attr)
    ppi = Data(x = x, edge_index=ppi_edge_index, edge_attr=ppi_edge_attr)

    store = ppi.edge_stores

    print(len(store), store[0].edge_index.shape, store[0].edge_attr.shape)# , ppi.edge_stores.edge_index, ppi.edge_stores.edge_attr)
    
    # Pruning edges by weight
    
    droplayer = DropLowWeightEdges(0.5)
    ppi = droplayer(ppi)

        