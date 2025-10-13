import sys
import os
from clustpy.metrics import unsupervised_clustering_accuracy as acc
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
import numpy as np 
import time
import torch
from clustpy.deep import IDEC, detect_device, DKM 
from KhatriRaoDeepClustering.khatrirao_dkm import KR_DKM 
from KhatriRaoDeepClustering.khatrirao_dec import KR_IDEC 
from run_experiments_utils import compute_inertia




def run_deep_clustering_experiment(algo_name,  n_clusters1, n_clusters2, had_ae, ae, nrep, batch_size): 
    '''
    Run deep clustering experiment 

    Parameters
    ----------
    algo_name : string (algorithm name)
    n_clusters1: int (number of protocentroids in the first set) 
    n_clusters2: int (number of protocentroids in the second set)
    had_ae: HadamardAutoencoder (compressed autoencoder used by Khatri-Rao clustering) 
    ae: FeedforwardAutoencoder (uncompressed (full) autoencoder used by standard clustering) 
    nrep: (int) number of initialzations 
    batch_size: (int) batch size for batch-wise optimization 


    Returns
    -------
    metrics: (dict) dictionary with clustering quality metrics 
    '''



    best_inertia = float("inf")
    best_ari, best_acc, best_nmi, best_time = None, None, None, None
    n_clusters_max = n_clusters1 * n_clusters2

    for i in range(nrep): 
        start_time = time.time() 

        if algo_name =="KR IDEC": 
            algo = KR_IDEC(
                n_clusters1, n_clusters2,
                neural_network=had_ae,
                random_state=i,
                batch_size=batch_size,
                clustering_loss_weight=1
            )
        elif algo_name == "IDEC"
            algo = IDEC(
                n_clusters_max,
                neural_network=ae,
                random_state=i,
                batch_size=batch_size,
                clustering_loss_weight=1.
            )
        elif algo_name == "KR DKM": 
            algo = KR_DKM(
                n_clusters1,
                n_clusters2,
                neural_network=had_ae,
                random_state=i,
                batch_size=batch_size,
                clustering_loss_weight=1
            )
        elif algo_name == "DKM": 
            algo = DKM(
            n_clusters_max,
            neural_network=ae,
            random_state=i,
            batch_size=batch_size,
            clustering_loss_weight=1.
            )
        else: 
            print("Wrong input dataset. Algorithm name must be one of KR IDEC, IDEC, KR DKM or DKM.")  
            sys.exit(0) 

        algo.fit(X)
        elapsed_time = time.time() - start_time 

        my_ari = ari(L, algo.labels_)
        my_acc = acc(L, algo.labels_)
        my_nmi = nmi(L, algo.labels_)

        # inertia monitoring 
        X_transformed = algo.transform(X) 
        inert = compute_inertia(X_transformed, algo.labels_, algo.cluster_centers_)
        if inert < best_inertia:
            best_inertia = inert
            best_ari, best_acc, best_nmi = my_ari, my_acc, my_nmi
            best_time = elapsed_time

    # results 
    metrics = {
        "ARI": best_ari,
        "ACC": best_acc,
        "NMI": best_nmi,
        "elapsed_time": best_time
    }

    return metrics