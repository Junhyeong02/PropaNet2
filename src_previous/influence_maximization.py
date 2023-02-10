import networkx as nx
import numpy as np
import pandas as pd

import myPropagation as NP
import src.Target_genes as TG 

def infByNode(TF, g, TFset) :
    visited = set()
    profit = 0
    stack = []
    stack.append(TF)
    while len(stack)>0:
        v = stack.pop()
        if v not in visited :
            visited.add(v)
            if v not in TFset: profit += abs(g.node[v]['weight'])
            for s in g.successors(v) :
                if s not in visited:
                    stack.append(s)
    visitedTGs = visited - TFset
    return profit, len(visited)-1, len(visitedTGs) 

def IM(nwk,TFset,repeat) :
    '''
    Influence Maximization
    ==============================
    Input 
        nwk (nx.object)
        TFset (set)
        repeat (int) 
    Output
        infNo (dict): reachability for each TF
        TFRank (list): sorted TF list'
    '''
    nodes=pd.Series(list(set([x for x in nwk.nodes()])))
    infNo = {}
    for n in TFset :
        infNo[n]=0.0
    for i in range(repeat) :
        # Produce G'
        g = nwk.copy()
        #for (a,b) in network.edges() :
        #    if np.random.randint(1,1000)>abs(g[a][b]['weight']*1000) :
        #        g.remove_edge(a, b)
                #Calculate influcence (profit)
        for TF in TFset :
            profit, lenInf, lenInfTG = infByNode(TF, g, TFset)
            #print TF,profit,lenInf,lenInfTG
            if lenInf>0 and not np.isnan(profit/float(lenInf)) : infNo[TF] += profit/float(lenInf)
    for n in TFset:
        infNo[n]=infNo[n]/float(repeat)
    TFRank = sorted(infNo.keys(), key=lambda x: infNo[x], reverse=True)
    for key in infNo.keys() :
        if (infNo[key]==0.0) : TFRank.remove(key)
    return TFRank, infNo

def TF_adding_NP(DEGli, geneSet, TFli, TFrankFile, DEGnetFile, seed, coverNo=200, coverage=None):
    '''
    Trim TF list with improving correlation using Network propagation    
    =================================================================
    Input
        DEGli (li)
        geneSet (set)
        TFli (li) 
        TFrankFile (str)
        DEGnetFile (str)
        seed (pd.DataFrame)
        coverNo (int)
        coverage (float)           
    Output
        spearmanLi (li)
        cover (li)
        TF_trimmed (li)
        lst_node_weight (li)
    '''
    DEGnet = nx.read_edgelist(DEGnetFile,data=(('weight',float),),create_using=nx.DiGraph())
    TFset=set(TFli)
    nodeCnt=len(set(DEGnet.nodes())-TFset)
    TFli_rank=pd.read_csv(TFrankFile,sep='\t',header=None)[0].tolist()
    corPre=-np.inf
    corr=[]
    cover=[]
    TF_trimmed=[]
    TG_set=set()
    prevCor = 0
    currTol = 0
    spearmanLi = []
    for iTF,TF in enumerate(TFli_rank):
        #seedFile: only selected TF( TFli[:iTF] ) is marked, otherwise 0
        TF_trimmed.append(TFli_rank[iTF])
        seed2weight = seed.loc[TF_trimmed,:].T.to_dict('records')[0]
        wk = NP.Walker(DEGnetFile, absWeight=True)
        spearman, lst_node_weight = wk.run_exp(seed2weight, TFset, 0.1, normalize=False)
        spearmanLi.append(spearman)
        corTmp=0
        corTmp=spearman[0]
        corr.append(corTmp)
        #TG_set |= TG.Target_genes(TF,DEGnet,DEGli,TFset)
        edges, TGalltmp, TGtmp = TG.Target_genes(TF,DEGnet,DEGli,TFset,geneSet)
        if TG_set == (TG_set | TGtmp): 
            TF_trimmed.pop()
            continue
        TG_set |= TGtmp
        # print '-----TF\t',TF_trimmed[-1]
        # print 'correlation\t%.3f' % corTmp
        # print 'len TG_set\t', len(TG_set)
        c=float(len(TG_set))/nodeCnt
        # print 'coverage\t%.3f' % c
        cover.append(c)
        if prevCor > corTmp : currTol +=1
        else : currTol = 0
        if coverage != None and (cover[-1] > coverage or len(TG_set)>coverNo):
            TF_trimmed.pop()
            break
        corPre = corTmp
    seed2weight = seed.loc[TF_trimmed,:].T.to_dict('records')[0]
    _, lst_node_weight = wk.run_exp(seed2weight, TFset, 0.1, normalize=False)
    return spearmanLi, cover, TF_trimmed, lst_node_weight
    