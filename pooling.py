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

def matPooled(adj): 
    id2group = []
    group2ids = {}
    for idx in range(adj.shape[0]): 
        group = int(idx/2)
        id2group.append(group)
        if not group in group2ids: 
            group2ids[group] = []
        group2ids[group].append(idx)
    size = int((adj.shape[0]-1)/2) + 1

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
        print("Please use: python3 pooling.py [adj filename]")
        print(" -> For example: python3 pooling.py data/metr-la/adj.npz")
        exit()

    fileAdj = sys.argv[1]
    adj = sp.load_npz(fileAdj)
    print("Adj loaded:", adj.shape, adj.dtype, adj.getformat(), np.sum(np.sum(adj)))
    adj = adj.toarray()

    pooled = matPooled(adj)
    print("Pooled matrix computed:", pooled.shape, np.sum(np.sum(pooled)))
    
    pooled = sp.csc_array(pooled)
    
    dirname = os.path.dirname(fileAdj)
    basename = os.path.basename(fileAdj)
    filePooled = os.path.join(dirname, basename[:-4]+"_p.npz")
    sp.save_npz(filePooled, pooled)

    print("Pooled matrix saved:", filePooled)


