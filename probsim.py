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
import scipy.stats as stats
from sklearn.cluster import * 

def matClustered(adj, vel): 
    minval = np.min(vel)
    maxval = np.max(vel)
    hists = []
    for idx in range(vel.shape[0]): 
        hist, bins = np.histogram(vel[idx], range=(minval, maxval))
        hist = hist / np.sum(hist)
        hists.append(hist)
    affine = np.zeros_like(adj)
    for idx in range(affine.shape[0]): 
        for jdx in range(affine.shape[1]): 
            affine[idx, jdx] = stats.entropy(hists[idx]+1e-3, hists[jdx]+1e-3)

    algo = AffinityPropagation(damping=0.5, max_iter=5)
    algo.fit(affine)
    ids = algo.labels_.tolist()

    id2group = []
    group2ids = {}
    for idx in range(adj.shape[0]): 
        group = ids[idx]
        id2group.append(group)
        if not group in group2ids: 
            group2ids[group] = []
        group2ids[group].append(idx)
    size = max(ids) + 1

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
    if len(sys.argv) < 3: 
        print("Please use: python3 probsim.py [adj filename] [vel filename]")
        print(" -> For example: python3 probsim.py data/metr-la/adj.npz data/metr-la/vel.csv")
        exit()

    fileAdj = sys.argv[1]
    adj = sp.load_npz(fileAdj)
    print("Adj loaded:", adj.shape, adj.dtype, adj.getformat())
    adj = adj.toarray()

    fileVel = sys.argv[2]
    vel = pd.read_csv(fileVel).to_numpy().T
    print("Vel loaded:", vel.shape, vel.dtype)

    clustered = matClustered(adj, vel)
    print("Clustered matrix computed:", clustered.shape, np.sum(np.sum(clustered)))
    
    clustered = sp.csc_array(clustered)
    
    dirname = os.path.dirname(fileAdj)
    basename = os.path.basename(fileAdj)
    fileClustered = os.path.join(dirname, basename[:-4]+"_k.npz")
    sp.save_npz(fileClustered, clustered)

    print("Clustered matrix saved:", fileClustered)


