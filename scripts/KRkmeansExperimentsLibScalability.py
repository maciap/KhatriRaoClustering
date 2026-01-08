import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy 
import numpy as np 
from sklearn.preprocessing import StandardScaler
import pickle
from collections import defaultdict
import time
import psutil
from KathriRaokMeans.kr_k_means_space_efficient import KrKMeans
from scipy.spatial.distance import cdist
import random
from KRkmeansExperimentsLib import makerealdata_v2, half_kron_rows, create_idx_map, coordinate_descent_half_kron




class StandardKMeansV2:
    """
    A standard K-Means implementation designed to mirror the structure of KrKMeans implementation.
    Vectorized distance computation, explicit update loop, convergence check, etc.
    """

    def __init__(self, X, k, standardize=True):
        """
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n, m)
        k : int
            Number of clusters
        standardize : bool
            Whether to z-score the input
        """
        if standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        self.X = X.astype("float32")
        self.k = k
        self.n, self.m = X.shape

        # slot that will hold centroids after initialization
        self.centroids = None

    def random_initialization(self):
        """Randomly pick k points from X as initial centroids."""
        idxs = np.random.choice(self.n, self.k, replace=False)
        return self.X[idxs].copy()

    def kmeans_plus_plus_init(self):
        """
        Deterministic k-means++ initialization,
        similar to your KrKMeansV2 implementation.
        """
        centroids = []
        # 1) choose first centroid randomly
        idx0 = random.randrange(self.n)
        centroids.append(self.X[idx0])

        # 2) iteratively pick farthest points
        for _ in range(1, self.k):
            # distances from all points to nearest existing centroid
            dists = cdist(self.X, np.vstack(centroids), metric="sqeuclidean")
            min_dist = np.min(dists, axis=1)
            next_idx = np.argmax(min_dist)
            centroids.append(self.X[next_idx])

        return np.vstack(centroids)

    def find_cluster_membership(self, X):
        """
        Computes closest centroid for each point.
        Returns labels, shape (n,)
        """
        dists = cdist(X, self.centroids, metric="sqeuclidean")
        return np.argmin(dists, axis=1)

    def update_centroids(self, labels):
        """
        Compute mean of points inside each cluster.
        """
        new_centroids = np.zeros((self.k, self.m), dtype="float32")

        for j in range(self.k):
            idxs = np.where(labels == j)[0]
            if len(idxs) > 0:
                new_centroids[j] = self.X[idxs].mean(axis=0)
            else:
                # empty cluster: reinitialize to random data point
                new_centroids[j] = self.X[np.random.randint(0, self.n)]

        return new_centroids

  
    def fit(self, n_iter=300, th_movement=1e-4, verbose=False, init_type="random"):
        """
        Standard Lloyd's K-Means clustering.

        Returns
        -------
        centroids : np.ndarray (k, m)
        labels : np.ndarray (n,)
        loss : float  SSE loss
        """

        # ---- Initialization phase ----
        if init_type == "random":
            self.centroids = self.random_initialization()
        elif init_type == "kmeans++":
            self.centroids = self.kmeans_plus_plus_init()
        else:
            print("Unknown init_type; defaulting to random.")
            self.centroids = self.random_initialization()

        for itr in range(n_iter):
            prev_centroids = self.centroids.copy()

            # --- ASSIGNMENT STEP ---
            labels = self.find_cluster_membership(self.X)

            # --- UPDATE STEP ---
            self.centroids = self.update_centroids(labels)

            # --- Convergence check ---
            movement = np.linalg.norm(self.centroids - prev_centroids)
            if verbose:
                print(f"iter {itr}, movement={movement:.6f}")
            if movement < th_movement:
                if verbose:
                    print(f"Converged after {itr} iterations.")
                break

        # final labels and loss
        labels = self.find_cluster_membership(self.X)
        diff = self.X - self.centroids[labels]
        loss = float(np.sum(diff**2))

        return self.centroids, loss, labels

   
   
    def predict(self, new_data):
        """
        Assign new data to clusters using final centroids.
        """
        dists = cdist(new_data, self.centroids, metric="sqeuclidean")
        return np.argmin(dists, axis=1)

            



class PP_KRkmeans(): 
    '''
    Implement post-processing-based naive Khatri-Rao clustering 

    Parameters
    ----------
    D : np.ndarray (input data matrix)
    h1 : int (cardinality first set of protocentroids)
    h2 : int (cardinality first set of protocentroids)
    standardize : bool, optional standardize the dataset before clustering (default: False)
    n_reps : int, optional
    lr : float, optional (learning rate)
    n_epochs : int, optional (epochs for gradient descent) 
    max_iter : int, optional (maximum number of iterations-default: 50) 

    '''
    def __init__(self, D, h1, h2, standardize = False, n_reps = 10, lr = 0.01, n_epochs = 1000, max_iter = 50):
        if standardize: 
            scaler = StandardScaler()
            D = scaler.fit_transform(D)
        self.D = D.astype('float32') 
        self.h1 = h1
        self.h2 = h2 
        self.n, self.m = D.shape
        self.lr = lr 
        self.n_epochs = n_epochs 
        self.max_iter = max_iter 
        self.n_reps = n_reps 

    def run_k_means(self): 
        ''' run k means ''' 

        KM = StandardKMeansV2(
        self.D,
        k=self.h1 * self.h2,
        standardize=False
        )

        l_lowes_bs = float("inf")
        bestcentroids = None
        best_assignments = None
        # --- multiple random restarts ---
        for _ in range(self.n_reps):
            centroids, l, assignments = KM.fit(
                n_iter=200,
                th_movement=0.0001,
                verbose=False,
                init_type="random",
            )
            if l < l_lowes_bs:
                l_lowes_bs = l
                bestcentroids = centroids
                best_assignments = assignments

        self.labels = best_assignments
        self.centroids = bestcentroids
        assert self.centroids.shape == (self.h1 * self.h2,self.m)
        

    def compute_loss_v2(self): 
        ''' compute inertia 
        Returns 
        -------
        inertia (float) 
        ''' 

        return np.sum(np.power((self.estimated_data - self.D),2))


    def decompose(self): 
        ''' decompose centroids into protocentroids

        Returns
        -------
        loss : float
        [B_estimate, B1, B2] : (list) list containing centroids and protocentroids 
        
        
         '''
        B_estimate, B1, B2 = coordinate_descent_half_kron(self.centroids, h1=self.h1, h2= self.h2,  T = self.n_epochs)
        self.estimated_centroids = B_estimate 
        
        # Compute squared Euclidean distances
        assignments = []
        for x in self.D:
            # Compute distances from this point to all centroids
            dists = np.linalg.norm(self.estimated_centroids - x, axis=1)
            # Assign to the closest centroid
            assignments.append(np.argmin(dists))

        assignments = np.array(assignments)
        self.labels = assignments
        self.estimated_data = self.estimated_centroids[self.labels]
        l = self.compute_loss_v2() 

        return l, [B_estimate, B1, B2]

    def fit(self): 
        ''' fit (run k means and decompose centroids into protocentroids)
        
        Returns
        -------
        loss : float
        Bs : list List containing centroids and protocentroids 
        self.labels : np.ndarray (final cluster assignments for each sample) 
        '''
        self.run_k_means() 
        loss, Bs = self.decompose() 
        return loss , Bs , self.labels 
        

def run_pp(X, n_clusters1, n_clusters_2,n_reps): 
    '''
    Run naive approach 

     Parameters
    ----------
    X : np.ndarray (input data matrix)
    n_clusters1 : int (cardinality first set of protocentroids)
    n_clusters_2 : int (cardinality second set of protocentroids)
    n_reps : int (number of repetitions for initialization) 

    Returns
    -------
    loss : float
    Bs : list List containing centroids and protocentroids 
    labels : np.ndarray (final cluster assignments for each sample) 


    '''
    pp = PP_KRkmeans(X, n_clusters1, n_clusters_2, n_reps=n_reps)
    loss, Bs, labels = pp.fit() 
    return loss, Bs, labels

def run_kr_k_means(
    X,
    n_clusters_1,
    n_clusters_2,
    n_reps=20,
    init_type="random",
    operator="sum",
    impl="original",
):
    """
    Run Kr Kmeans clustering.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix.
    n_clusters_1 : int
        Cardinality of first set of protocentroids.
    n_clusters_2 : int
        Cardinality of second set of protocentroids.
    n_reps : int
        Number of repetitions (random restarts) for initialization.
    init_type : str
        Type of initialization passed to KrKMeans.fit
        ("random", "kmeans++ random", "kmeans++ deterministic", or [B1,B2] list).
    operator : {"sum", "product"}
        Aggregator function for combining protocentroids.
   
    Returns
    -------
    l_lowes_bs : float
        Best (lowest) inertia over the n_reps runs.
    [B_1, B_2] : list of np.ndarray
        Best protocentroids.
    best_assignments : np.ndarray
        Final cluster assignments for each sample (flat index in [0, h1*h2-1]).
    """

    
    # --- construct model ---
    dKM = KrKMeans(
        X,
        h1=n_clusters_1,
        h2=n_clusters_2,
        standardize=False,
        operator=operator,
    )

    l_lowes_bs = float("inf")
    bestABs = None
    best_assignments = None

    # --- multiple random restarts ---
    for _ in range(n_reps):
        ABs, l, assignments = dKM.fit(
            n_iter=200,
            th_movement=0.0001,
            verbose=False,
            init_type=init_type,
        )
        if l < l_lowes_bs:
            l_lowes_bs = l
            bestABs = ABs
            best_assignments = assignments

    A_1, B_1, A_2, B_2 = bestABs
    return l_lowes_bs, [B_1, B_2], best_assignments


def run_k_means_numpy(
    X,
    n_clusters,
    n_reps=20,
    init_type="random"
            ):
    """
    Run Kr Kmeans clustering using a chosen implementation.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix.
    n_clusters_1 : int
        Cardinality of first set of protocentroids.
    n_clusters_2 : int
        Cardinality of second set of protocentroids.
    n_reps : int
        Number of repetitions (random restarts) for initialization.
    init_type : str
        Type of initialization passed to KrKMeans.fit
        ("random", "kmeans++ random", "kmeans++ deterministic", or [B1,B2] list).
    operator : {"sum", "product"}
        Aggregator function for combining protocentroids.
    impl : {"original", "optimized_no_faiss", "optimized_faiss"}
        Which KrKMeans implementation to use.

    Returns
    -------
    l_lowes_bs : float
        Best (lowest) inertia over the n_reps runs.
    [B_1, B_2] : list of np.ndarray
        Best protocentroids.
    best_assignments : np.ndarray
        Final cluster assignments for each sample (flat index in [0, h1*h2-1]).
    """


    # --- construct model ---
    KM = StandardKMeansV2(
        X,
        k=n_clusters,
        standardize=False
        )

    l_lowes_bs = float("inf")
    bestcentroids = None
    best_assignments = None
    # --- multiple random restarts ---
    for _ in range(n_reps):
        centroids, l, assignments = KM.fit(
            n_iter=200,
            th_movement=0.0001,
            verbose=False,
            init_type=init_type,
        )
        if l < l_lowes_bs:
            l_lowes_bs = l
            bestcentroids = centroids
            best_assignments = assignments

    return l_lowes_bs, bestcentroids, best_assignments
