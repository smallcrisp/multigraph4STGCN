'''
author: smallcrisp
-----
input: the adjacency graph in .npz format
output: the coarsened graph, but in the same size (multi-to-one)
'''
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import * 

def matClustered(adj): 
    nclusters = round(adj.shape[0]/2)
    algo = KMeans(n_clusters=nclusters, n_init='auto')
    algo.fit(adj)
    ids = algo.labels_.tolist()

    id2group = []
    group2ids = {}
    for idx in range(adj.shape[0]): 
        group = ids[idx]
        id2group.append(group)
        if not group in group2ids: 
            group2ids[group] = []
        group2ids[group].append(idx)
    size = nclusters

    small = np.zeros((size, size))
    for idx in range(size): 
        for jdx in range(size): 
            total = 0
            count = 0
            for kdx in group2ids[idx]: 
                for ldx in group2ids[jdx]: 
                    if adj[kdx, ldx] > 0: 
                        total += adj[kdx, ldx]
                        count += 1
            average = total / max(count, 1e-5)
            small[idx, jdx] = average

    result = np.zeros_like(adj)
    for idx in range(adj.shape[0]): 
        for jdx in range(adj.shape[1]): 
            result[idx, jdx] = small[id2group[idx], id2group[jdx]]

    return result

if __name__ == "__main__": 
    if len(sys.argv) < 2: 
        print("Please use: python3 clustering.py [adj filename]")
        print(" -> For example: python3 clustering.py data/metr-la/adj.npz")
        exit()

    fileAdj = sys.argv[1]
    adj = sp.load_npz(fileAdj)
    print("Adj loaded:", adj.shape, adj.dtype, adj.getformat(), np.sum(np.sum(adj)))
    adj = adj.toarray()

    clustered = matClustered(adj)
    print("Clustered matrix computed:", clustered.shape, np.sum(np.sum(clustered)))
    
    clustered = sp.csc_array(clustered)
    
    dirname = os.path.dirname(fileAdj)
    basename = os.path.basename(fileAdj)
    fileClustered = os.path.join(dirname, basename[:-4]+"_c.npz")
    sp.save_npz(fileClustered, clustered)

    print("Clustered matrix saved:", fileClustered)


