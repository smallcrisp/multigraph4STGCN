'''
author: smallcrisp

input: the adjacency graph in .npz format and the velocity matrix in csv format
output: the distance, neighboring, functionality, heuristics, similarity graph in .npz format
'''
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.optimize as opt

def matDistance(adj): 
    return adj

def matNeighbor(adj): 
    return (adj > 0)

def matFunction(adj): 
    return np.corrcoef(adj)

def matHeuristics(adj, vel, sigma=1): 
    def f(x, a, b): 
        return a * np.exp(-b * x)
    assert adj.shape[0] == vel.shape[0]
    minval = np.min(vel)
    maxval = np.max(vel)
    matA = np.zeros(vel.shape[0])
    matB = np.zeros(vel.shape[0])
    for idx in range(vel.shape[0]): 
        hist, bins = np.histogram(vel[idx], range=(minval, maxval))
        hist = hist / np.sum(hist)
        popt, pcov = opt.curve_fit(f, np.arange(0, len(hist)), hist)
        matA[idx] = popt[0]
        matB[idx] = popt[1]
    matD = np.sqrt((matA[:, None] - matA[None, :]) ** 2 + (matB[:, None] - matB[None, :]) ** 2)
    matW = np.exp(-matD / sigma)
    for idx in range(vel.shape[0]): 
        matW[idx, idx] = 0
    return matW

def matSimilarity(vel): 
    return np.corrcoef(vel)

if __name__ == "__main__": 
    if len(sys.argv) < 3: 
        print("Please use: python3 multigraph.py [adj filename] [vel filename]")
        print(" -> For example: python3 multigraph.py data/metr-la/adj.npz data/metr-la/vel.csv")
        exit()

    fileAdj = sys.argv[1]
    adj = sp.load_npz(fileAdj)
    print("Adj loaded:", adj.shape, adj.dtype, adj.getformat())
    adj = adj.toarray()

    fileVel = sys.argv[2]
    vel = pd.read_csv(fileVel).to_numpy().T
    print("Vel loaded:", vel.shape, vel.dtype)

    dist = matDistance(adj)
    neigh = matNeighbor(adj)
    funcs = matFunction(adj)
    heur = matHeuristics(adj, vel)
    sim = matSimilarity(vel)

    print("Distance matrix computed:", dist.shape, np.sum(np.sum(dist)))
    print("Neighboring matrix computed:", neigh.shape, np.sum(np.sum(neigh)))
    print("Functionality matrix computed:", funcs.shape, np.sum(np.sum(funcs)))
    print("Heuristic matrix computed:", heur.shape, np.sum(np.sum(heur)))
    print("Similarity matrix computed:", sim.shape, np.sum(np.sum(sim)))

    dist = sp.csc_array(dist)
    neigh = sp.csc_array(neigh)
    funcs = sp.csc_array(funcs)
    heur = sp.csc_array(heur)
    sim = sp.csc_array(sim)
    
    dirname = os.path.dirname(fileAdj)
    fileDistance = os.path.join(dirname, "dist.npz")
    fileNeighbor = os.path.join(dirname, "neigh.npz")
    fileFunction = os.path.join(dirname, "funcs.npz")
    fileHeuristics = os.path.join(dirname, "heur.npz")
    fileSimilarity = os.path.join(dirname, "sim.npz")
    sp.save_npz(fileDistance, dist)
    sp.save_npz(fileNeighbor, neigh)
    sp.save_npz(fileFunction, funcs)
    sp.save_npz(fileHeuristics, heur)
    sp.save_npz(fileSimilarity, sim)

    print("Distance matrix saved:", fileDistance)
    print("Neighboring matrix saved:", fileNeighbor)
    print("Functionality matrix saved:", fileFunction)
    print("Heuristic matrix saved:", fileHeuristics)
    print("Similarity matrix saved:", fileSimilarity)
    
