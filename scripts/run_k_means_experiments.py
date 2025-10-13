import KRkmneansExperimentsLib
import sys 
import os
import numpy as np 
import pickle 
import time
import clustbench
from clustpy.deep import IDEC, detect_device
from clustpy.data import load_optdigits, load_mnist, load_letterrecognition, z_normalization
from clustpy.metrics import unsupervised_clustering_accuracy as acc
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from clustpy.utils import plot_with_transformation
from run_experiments_utils import (stratified_subsample, closest_factors, concatenate_mnist, generate_blobs, 
                                  generate_swiss_roll, generate_classification, shift_to_zero, load_data)


if __name__ == "__main__": 

    dataset = sys.argv[1] 
    random_state = 42
    batch_size = 512
    n_reps = 20  
    
    X,L = load_data(dataset)

    L = shift_to_zero(L) # check that the minimum is zero otherwise translate 

    data_path = "data/clustering-data-v1"
    n_labels = len(np.unique(L))
    n_clusters1, n_clusters2 = closest_factors(n_labels)

    if dataset!="stickfigures" and dataset!="mnist" and dataset!="double_mnist": 
        mean = np.mean(X, axis=0)        
        std = np.std(X, axis=0)          
        # add epsilon to avoid division by zero
        X = (X - mean) / (std + 1e-8)
    else: 
        X = X / np.max(X)

    n_clusters_sum = n_clusters1 + n_clusters2
    n_clusters_max = n_clusters1 * n_clusters2
    timings = {}

    # post processing naive KR clustering 
    start = time.time()
    loss_pp, Bs_pp, labels_pp = KRkmneansExperimentsLib.run_pp(X, n_clusters1, n_clusters2, n_reps=n_reps)
    labels_pp = shift_to_zero(labels_pp) 
    end = time.time()
    metrics_pp = KRkmneansExperimentsLib.compute_metrics(labels_pp, L)
    metrics_pp["inertia"] = loss_pp
    timings["pp"] = end - start


    # k means with same centroid parameters (h1 + h2) 
    start = time.time()
    loss, centroids, labels = KRkmneansExperimentsLib.run_k_means_baseline(X, n_clusters_sum, n_reps=n_reps) 
    labels = shift_to_zero(labels) 
    end = time.time()
    metrics_km = KRkmneansExperimentsLib.compute_metrics(labels, L)
    metrics_km["inertia"] = loss
    timings["kmeans"] = end - start


    # k means with more centroid parameters (h1 * h2) 
    start = time.time()
    loss_max, centroids_max, labels_max = KRkmneansExperimentsLib.run_k_means_baseline(X, n_clusters_max, n_reps=n_reps) 
    labels_max = shift_to_zero(labels_max) 
    end = time.time()
    metrics_km_max = KRkmneansExperimentsLib.compute_metrics(labels_max, L)
    metrics_km_max["inertia"] = loss_max
    timings["kmeans"] = end - start


    # KR clustering with sum aggregator 
    start = time.time()
    l_lowes_kr_sum, Bs_kr_sum, best_assignments_sum = KRkmneansExperimentsLib.run_kr_k_means(X, n_clusters1, n_clusters2, n_reps=n_reps, init_type = "random", operator = "sum") 
    best_assignments_sum = shift_to_zero(best_assignments_sum) 
    end = time.time()
    metrics_kr_sum = KRkmneansExperimentsLib.compute_metrics(best_assignments_sum, L)
    metrics_kr_sum["inertia"] = l_lowes_kr_sum
    timings["kr_sum"] = end - start

    
    # KR clustering with product aggregator  
    start = time.time()
    l_lowes_kr_prod, Bs_kr_prod , best_assignments_kr_prod = KRkmneansExperimentsLib.run_kr_k_means(X, n_clusters1, n_clusters2, n_reps=n_reps, init_type = "random", operator = "product") 
    best_assignments_kr_prod = shift_to_zero(best_assignments_kr_prod) 
    end = time.time()
    metrics_kr_prod = KRkmneansExperimentsLib.compute_metrics(best_assignments_kr_prod, L)
    metrics_kr_prod["inertia"] = l_lowes_kr_prod
    timings["kr_prod"] = end - start