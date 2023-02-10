import pandas as pd
import argparse
import numpy as np
from scipy.stats import pearsonr
import multiprocessing

def myCorr(x):
    g1, g2 = sorted(x)
    if g1 == g2:
        val = 0.0
    else:
        val, pval = pearsonr(lst_exps[g1], lst_exps[g2])
    return (g1, g2, val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='python %(prog)s -nwk nwkFile -exp expFile -o out')
    parser.add_argument('-nwk', required=True, help='Network file')
    parser.add_argument('-exp', required=True, help='gene expression File')
    parser.add_argument('-p', type=int, metavar='N',
                        default='1', help='total process')
    parser.add_argument('-o', required=True, help='out File')
    args = parser.parse_args()

    gene_exp = pd.read_csv(args.exp, sep="\t")
    gene_nwk = pd.read_csv(args.nwk, sep="\t", names=["start", "end"])

    print(gene_exp.shape)
    print(gene_nwk.shape)

    gene_set = set(gene_exp["gene"])
    gene_nwk_filtered = gene_nwk[[start in gene_set and end in gene_set for start, end in zip(
        gene_nwk["start"], gene_nwk["end"])]]

    print(gene_nwk_filtered.shape)

    pool = multiprocessing.Pool(args.p)
    res = pool.imap_unordered(myCorr, lst_pairs)

    with open(args.o, 'w') as f3:
        for g1, g2, val in res:
            if g1 == g2:
                continue
            f3.write('\t'.join([g1, g2, str(val)])+'\n')

    print('network done')
