import pandas as pd
import numpy as np
from copy import deepcopy
import os
import argparse
import sys
import networkx as nx
import time
import myPropagation as NP

import multiprocessing 
from functools import partial
from collections import Counter

from influence_maximization import IM, TF_adding_NP

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('TFliFile',help='TF list File')
    parser.add_argument('nwkFile',help='Network file')
    parser.add_argument('expFile',help='gene expression file with z-value,seed file for NP')
    parser.add_argument('binFile',help='binary file of condition from DEGlimma')
    #parser.add_argument('binCtrl',help='binary file of Control data from DEGlimma')
    parser.add_argument('-geneSet',help='gene set by user') 
    parser.add_argument('-cond',required=True,help='condition name') 
    parser.add_argument('-outD',help='project (output directory) name')
#    parser.add_argument('-r',help='# repeats for IM')
    parser.add_argument('-p',default='10',type=int,help='# process for multiprocessing')
    parser.add_argument('-c',default='0.5',type=float,help='coverage threshold')
    parser.add_argument('-coverNo',default='200',type=float)
    args=parser.parse_args()

    if args.outD != None:
        if not os.path.exists(args.outD) : os.mkdir(args.outD)
    exp = pd.read_csv(args.expFile, sep='\t',index_col=0)
    with open(args.TFliFile) as tfF:
        TFli=tfF.read().strip().split()
   
    #networkFile --> networkx object, expFile --> pd.DataFrame
    print('TF length', len(set(TFli)))
    n=len(exp.columns)
    network = nx.read_edgelist(args.nwkFile,data=(('weight',float),),create_using=nx.DiGraph()) 
    bins = pd.read_csv(args.binFile, sep='\t',index_col=0)
    #bins_ctrl = pd.read_csv(args.binCtrl, sep='\t',index_col=0)
    weight = (exp*(bins.applymap(lambda x:abs(x))))
    #IM, NP for each timepoint
    def main_fxn(i):
        #step0: parsing exp data, network for each timepoint
        start=time.time() 
        ##Get gene expression data for specific timepoint 
        #weight_tp_cond = exp.iloc[:,i-1]
        weight_tp = exp.iloc[:,i-1].reset_index()
        ##exclude DEGs from control
        #DEG_tp_ctrl = bins_ctrl.loc[bins_ctrl.iloc[:,i-1]==1|-1].index 
        #weight_tp = weight_tp_cond.loc[weight_tp_cond.index.difference(DEG_tp_ctrl)].reset_index()
        weight_tp.columns = ['gene','weight']
        ##Get DEG list
        #DEG_tp_cond = weight.iloc[:,i-1]
        DEG_tp =  weight.iloc[:,i-1].reset_index()
        #DEG_tp = DEG_tp_cond.loc[DEG_tp_cond.index.difference(DEG_tp_ctrl)].reset_index()
        DEG_tp.columns = ['gene','weight']
        DEGli = DEG_tp[DEG_tp['weight']!=0]['gene'].tolist()
         
        DEGliFile =args.outD+'/'+ args.cond+'.DEGli.t'+str(i)
        
        with open(DEGliFile,'w') as f:
            f.write('\n'.join(DEGli))
        if args.geneSet != None:
            geneSet = set(np.genfromtxt(args.geneSet, dtype=np.str))&DEGli
        else:
            geneSet = set(DEGli)
        ##Make nwk with DEGs in specific timepoint
        
        nwk_tp = network.subgraph(geneSet|set(TFli))
        DEGnetFile = args.outD + '/' + args.cond + '.nwk.t'+ str(i) 
        nx.write_edgelist(nwk_tp,DEGnetFile,data=['weight'],delimiter='\t')
        ##Assign new weight to timepoint-specific network
        weight_DEG_nwk = weight_tp[weight_tp.gene.isin(nwk_tp.nodes())].set_index('gene')
        weight_dict = weight_DEG_nwk.T.to_dict('records')[0]
        nx.set_node_attributes(nwk_tp,weight_dict,'weight')
        # print >> sys.stderr,'tp', i,'network done',time.strftime("%H:%M:%S",time.gmtime(time.time()-start)),'\n'
        
        #step1: Influence maximization
        start=time.time() 
        ##preprocess
        nodes=pd.Series(list(set([x for x in nwk_tp.nodes()])))
        TFset=set(nodes[nodes.isin(TFli)].tolist())
        ##influence maximization
        TFrank, infNo = IM(nwk_tp,TFset,1)
        ##Output into file 
        TFrankFile =args.outD+'/'+ args.cond+'.TF_rank.t'+str(i)        
        with open(TFrankFile,'w') as TFrankF:
            for TF in TFrank:
                TFrankF.write(TF+'\n')#IM result--> cold.D2.TF_rank.1
        # with open(args.outD+'/'+args.cond+'.TFinf.t'+str(i), 'w') as infF:
        #     for TF in infNo : infF.write(TF+'\t'+str(infNo[TF])+'\n')   
        print >> sys.stderr,'time point', i,'TF ranking done', time.strftime("%H:%M:%S",time.gmtime(time.time()-start)), '\n'
    
        
        #step2: NP based on TF rank
        start=time.time() 
        ##TF adding NP
        seed = weight_tp.set_index('gene')
        spearmanLi,coverage,TF_trimmed,lst_node_weight = TF_adding_NP(DEGli, geneSet, TFli, TFrankFile, DEGnetFile, seed, args.coverNo, coverage=args.c) 
        ##Save Output as file 
        TFtrimF = TFrankFile + '.trim'

        with open(TFtrimF,'w') as f:
            f.write('\n'.join(TF_trimmed))
        
        dic_node2weights = {}
        column_name = exp.columns.tolist()
        lst_seed = exp.index.values.tolist()
        wk = NP.Walker(DEGnetFile, absWeight=True)
        set_nodes = set(wk.OG.nodes())
        set_tmpNodes = set()
        
        for node, weight_np, all_weight in lst_node_weight:
            if node not in dic_node2weights:
                dic_node2weights[node]=[]

            dic_node2weights[node].append(weight_np)
            set_tmpNodes.add(node)

        for node in set_nodes-set_tmpNodes:
            if node not in dic_node2weights:
                dic_node2weights[node]=[]

            dic_node2weights[node].append(0.0)
        # with open(args.outD+'/'+args.cond+'.NPresult.t'+str(i),'w') as OF:
        #     OF.write('Gene\t{}\n'.format(column_name[i-1]))
        #     for node, weights in dic_node2weights.items():
        #         OF.write('\t'.join(map(str,[node]+weights))+'\n')
        #         OF.flush()
        # with open(args.outD+'/log.corr.'+args.cond+'.t'+str(i),'w') as OF2:
        #     for item in spearmanLi:
        #         OF2.write('{}\t{}\n'.format(item[0],item[1]))
        #         OF2.flush()
        # print >> sys.stderr,'tp', i,'NP done', time.strftime("%H:%M:%S",time.gmtime(time.time()-start)), '\n'
        ###log
        #with open(args.outD+'/log.cover.'+args.cond+'.t'+ str(i),'w') as f2:
        #    for item in coverage:
        #        print >> f2, item
        return
    pool = multiprocessing.Pool(args.p)
    nums=range(1,n+1)
    pool.imap_unordered(main_fxn,nums)

    pool.close()
    pool.join()
