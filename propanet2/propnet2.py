import argparse
import sys
import os

import numpy as np
import pandas as pd

from torch_geometric.data import Data

from grn.network import GRNnetwork
from grn.build import edgelist_to_adjmatrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='python %(prog)s -nwk nwkFile -exp expFile -o out')
    parser.add_argument('-nwk', required=True, help='Network file')
    parser.add_argument('-exp', required=True, help='gene expression File')
    parser.add_argument('-p', type=int, metavar='N',
                        default='1', help='total process')
    parser.add_argument('-o', required=True, help='out File')
    args = parser.parse_args()

    network_edge = args.nwk
    gene_exp = pd.read_csv(args.exp, sep = "\t")

    tgt_list = list(gene_exp["gene"])
    adj_matrix = edgelist_to_adjmatrix(network_edge, tgt_list)

    