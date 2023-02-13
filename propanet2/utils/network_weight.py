import argparse
import sys
import os

import pandas as pd
import numpy as np

from tqdm import tqdm

from scipy.stats import pearsonr
from multiprocessing import Pool, Manager

def calculate_corr(x):
    q, g1, g2, exp1, exp2, corr_function = x
    corr, _ = corr_function(exp1, exp2)
    
    q.put((g1, g2, corr))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='python %(prog)s -nwk nwkFile -exp expFile -o out')
    parser.add_argument('-nwk', required=True, help='Network file')
    parser.add_argument('-exp', required=True, help='gene expression File')
    parser.add_argument('-o', required=True, help = "output file")
    args = parser.parse_args()

    with open(args.nwk) as f:
        edge_list = list(map(lambda x: x.strip().split(), f.readlines()))

    m = Manager()
    q = m.Queue()
    gene_exp = pd.read_csv(args.exp, sep = "\t", index_col=0)
    args_list = []

    corr_function = pearsonr

    print("prepare argument list...")
    
    nodata = set()

    for start, end in edge_list:
        try:
            args_list.append((q, start, end, gene_exp.loc[start, :], gene_exp.loc[end, :], corr_function))
        except KeyError:
            if start not in gene_exp.index:
                nodata.add(start)
            if end not in gene_exp.index:
                nodata.add(end)
            continue
            
    print(len(nodata))
    print("calculate correlation coefficent...")

    print(len(args_list))
    with Pool(processes=os.cpu_count()) as pool:
        with tqdm(total=len(args_list)) as pbar:
            for _ in pool.imap_unordered(calculate_corr, args_list):
                pbar.update()

    with open(args.o, 'w') as fw:
        while not q.empty():
            start, end, corr = q.get()
            fw.write('\t'.join([start, end, str(corr)])+'\n')

    print('network done')
