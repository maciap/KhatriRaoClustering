import time
from memory_profiler import memory_usage
import KRkmeansExperimentsLibScalability
import sys
import os
import numpy as np
import pickle
import time
import clustbench
from clustpy.metrics import unsupervised_clustering_accuracy as acc
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from run_experiments_utils import (
    closest_factors,
    generate_blobs, generate_classification,
    shift_to_zero
)
from sklearn.datasets import make_blobs, make_classification
import time



def run_with_time_and_memory(func, *args, **kwargs):
    """
    Run func(*args, **kwargs) and measure:
      - wall-clock runtime (seconds)
      - peak memory usage during the call (MiB)

    Returns
    -------
    result : any
        Return value of func.
    runtime : float
        Wall-clock time in seconds.
    mem_info : dict
        {
          "mem_min": float (MiB),
          "mem_max": float (MiB),
          "mem_peak_delta": float (MiB, max - min),
          "mem_trace": list[float] (MiB samples over time),
        }
    """
    def _wrapped():
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        t1 = time.perf_counter()
        return out, (t1 - t0)

    mem_trace, (result, runtime) = memory_usage(
        (_wrapped, (), {}),
        retval=True,
        interval=0.1,   # sampling interval in seconds
        timeout=None
    )

    mem_min = min(mem_trace)
    mem_max = max(mem_trace)
    mem_info = {
        "mem_min": mem_min,
        "mem_max": mem_max,
        "mem_peak_delta": mem_max - mem_min,
        "mem_trace": mem_trace,
    }
    return result, runtime, mem_info


def generate_blobs(n_samples, n_clusters, n_features):

    

    '''
    Generate Blobs dataset

  
    Returns
    -------
    X : np.ndarray (data) 
    L : np.ndarray (labels) 
    '''


    X, L = make_blobs(n_samples=n_samples, centers=n_clusters, n_features=n_features, random_state=42)
    return X, L


def generate_classification(n_samples, n_clusters, n_features):
    '''
    Generate Classification dataset

  
    Returns
    -------
    X : np.ndarray (data) 
    L : np.ndarray (labels) 
    '''

    X, L = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_clusters,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, L




if __name__ == "__main__":

    if len(sys.argv) < 3:
        raise SystemExit("Usage: python script.py <dataset> <to_vary>, what to_vary ∈ {n_points, n_features, n_centroids}")


    dataset = sys.argv[1] 
    to_vary = sys.argv[2]


    allowed = {"n_points", "n_features", "n_centroids"}
    if to_vary not in allowed:
        raise SystemExit(f"Invalid to_vary='{to_vary}'. Must be one of {sorted(allowed)}")


    random_state = 42
    batch_size = 512
    n_reps = 20  
    
    n_features = 100 
    n_clusters = 100


    post_processing_times = [] 
    kmeans_times = [] 
    kmeans_max_times = [] 
    kr_clustering_sum = []
    kr_clustering_prod = [] 

    if to_vary == "n_points": 
        n_features = 100 
        n_clusters = 100
        to_iterate = [4000, 8000, 12000, 16000]

    if to_vary == "n_features":
        n_points = 1000 
        n_clusters = 100 
        to_iterate = [5000, 10000, 15000, 20000]

    if to_vary == "n_clusters":
        n_points = 20000
        n_features = 100 
        to_iterate = [2500, 5000, 7500, 10000]

    
    for ite in to_iterate: 

        if dataset == "blobs":
            if to_vary == "n_points":
                X, L = generate_blobs(ite, n_clusters, n_features) 
            if to_vary == "n_features":
                X, L = generate_blobs(n_points, n_clusters, ite)
            if to_vary == "n_clusters":
                X, L = generate_blobs(n_points, ite, n_features)

        if dataset == "classification":
            if to_vary == "n_points":
                X, L = generate_classification(ite, n_clusters, n_features) 
            if to_vary == "n_features":
                X, L = generate_classification(n_points, n_clusters, ite)
            if to_vary == "n_clusters":
                X, L = generate_classification(n_points, ite, n_features)


        L = shift_to_zero(L)

        n_labels = len(np.unique(L))
        n_clusters1, n_clusters2 = closest_factors(n_labels)

        if dataset not in ["stickfigures", "mnist", "double_mnist"]:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            X = (X - mean) / (std + 1e-8)
        else:
            X = X / np.max(X)

        n_clusters_sum = n_clusters1 + n_clusters2
        n_clusters_max = n_clusters1 * n_clusters2

        timings = {}
        memories = {}

        # Post-processing naive KR clustering
        start_time = time.time() 
        (loss_pp, Bs_pp, labels_pp), t_pp, mem_pp = run_with_time_and_memory(
            KRkmeansExperimentsLibScalability.run_pp,
            X, n_clusters1, n_clusters2, n_reps=n_reps
        )
        timings["pp"] = t_pp
        memories["pp"] = mem_pp
        post_processing_times.append( (time.time() - start_time) ) 


        #  KMeans baseline with (r1 + r2) centroids
        start_time = time.time() 
        (loss, centroids, labels), t_km, mem_km = run_with_time_and_memory(
            KRkmeansExperimentsLibScalability.run_k_means_numpy,
            X, n_clusters_sum, n_reps=n_reps
        )
        timings["kmeans_sum"] = t_km
        memories["kmeans_sum"] = mem_km
        kmeans_times.append( (time.time() - start_time) / 20 ) 


        #  KMeans baseline with (r1 * r2) centroids
        start_time = time.time() 
        (loss_max, centroids_max, labels_max), t_km_max, mem_km_max = run_with_time_and_memory(
            KRkmeansExperimentsLibScalability.run_k_means_numpy,
            X, n_clusters_max, n_reps=n_reps
        )
        timings["kmeans_prod"] = t_km_max
        memories["kmeans_prod"] = mem_km_max
        kmeans_max_times.append( (time.time() - start_time) / 20 ) 


        # KR clustering with sum aggregator 
        start_time = time.time() 
        (l_lowes_kr_sum_opt, Bs_kr_sum_opt, best_assignments_sum_opt), t_kr_sum_opt, mem_kr_sum_opt = run_with_time_and_memory(
            KRkmeansExperimentsLibScalability.run_kr_k_means,
            X, n_clusters1, n_clusters2,
            n_reps=n_reps,
            init_type="random",
            operator="sum",
            impl="optimized_no_faiss",
        )
        timings["kr_sum_opt"] = t_kr_sum_opt
        memories["kr_sum_opt"] = mem_kr_sum_opt
        kr_clustering_sum.append( (time.time() - start_time) / 20 ) 


        # KR clustering with product aggregator 
        start_time = time.time() 
        (l_lowes_kr_prod_opt, Bs_kr_prod_opt, best_assignments_kr_prod_opt), t_kr_prod_opt, mem_kr_prod_opt = run_with_time_and_memory(
            KRkmeansExperimentsLibScalability.run_kr_k_means,
            X, n_clusters1, n_clusters2,
            n_reps=n_reps,
            init_type="random",
            operator="product",
            impl="optimized_no_faiss",
        )
        timings["kr_prod_opt"] = t_kr_prod_opt
        memories["kr_prod_opt"] = mem_kr_prod_opt
        kr_clustering_prod.append( (time.time() - start_time) / 20 )

