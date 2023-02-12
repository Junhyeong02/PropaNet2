import argparse
import numpy as np

from typing import List, Tuple, Optional

def create_adjmatrix(edge_list: List[Tuple[str, str, Optional[float]]], tgt_node_list: List[str], weighted = True)->np.array:
    adj_dict:dict = dict()
    tgt_node_set:set = set(tgt_node_list)

    for start_node in tgt_node_list:
        if start_node not in adj_dict:
            adj_dict[start_node] = dict()

        for end_node in tgt_node_list:
            adj_dict[start_node][end_node] = 0

    if weighted:
        for start, end, weight in edge_list:
            if start in tgt_node_set and end in tgt_node_set:
                adj_dict[start][end] = weight
    else:
        for start, end in edge_list:
            if start in tgt_node_set and end in tgt_node_set:
                adj_dict[start][end] = 1

    adj_matrix:list = list()
    
    for start_node in tgt_node_list:
        adj_matrix.append(list(adj_dict[start_node][end_node] for end_node in tgt_node_list))

    adj_matrix:np.array = np.array(adj_matrix)
    
    return adj_matrix

if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-nwk", required = True, help = "Network file, node pair of each edges")
    parser.add_argument("-tgt", required = True, help = "List of target node")

    args = parser.parse_args()
    network_file = args.nwk
    tgt_node_file = args.tgt
    
    with open(network_file) as f:
        edge_list = list(map(lambda x: x.strip().split(), f.readlines()))
    
    with open(tgt_node_file) as f:
        tgt_node_list = list(map(lambda x: x.strip(), f.readlines()))

    adj_matrix = create_adjmatrix(edge_list, tgt_node_list, weighted = False)
    
    print(adj_matrix.shape, len(tgt_node_list))
        
