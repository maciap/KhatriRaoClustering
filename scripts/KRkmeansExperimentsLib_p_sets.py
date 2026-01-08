import copy 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from clustpy.metrics import unsupervised_clustering_accuracy as acc
import os
import pickle
from collections import defaultdict
import os
import time
import psutil
from KathriRaokMeans.kr_k_means_p_sets import KrKMeans
from KRkmeansExperimentsLib import makerealdata_v2

        
def hadamard_cartesian_rows(B_list):
    """
    Generalization of `half_kron_rows` to p factors.

    Given a list of p matrices [B0, B1, ..., B_{p-1}],
    where Bl has shape (r_l, m), return a matrix of shape
    (prod_l r_l, m) where each row is the elementwise product
    of one row from each Bl.

    Parameters
    ----------
    B_list : list of np.ndarray
        List of factor matrices, each of shape (r_l, m).

    Returns
    -------
    out : np.ndarray
        Array of shape (prod_l r_l, m). Each row corresponds to
        an elementwise product of rows picked from each factor.
    """
    p = len(B_list)
    m = B_list[0].shape[1]
    ranks = [B.shape[0] for B in B_list]
    total_rows = int(np.prod(ranks))

    out = np.empty((total_rows, m), dtype=B_list[0].dtype)

    # Multi-index over all combinations of row indices
    cnt = 0
    for multi_idx in np.ndindex(*ranks):
        row_prod = np.ones(m, dtype=B_list[0].dtype)
        for l, idx_l in enumerate(multi_idx):
            row_prod *= B_list[l][idx_l, :]
        out[cnt, :] = row_prod
        cnt += 1

    return out




def coordinate_descent_multi_hadamard(B_centroids, ranks, T=5000):
    """
    Coordinate descent to factorize B_centroids into p Hadamard factors.

    We assume:
        B_centroids.shape = (prod_l ranks[l], m)
        B_centroids[unravel(i0,...,i_{p-1}), :] ≈ prod_l B_list[l][i_l, :]

    Parameters
    ----------
    B_centroids : np.ndarray, shape (prod ranks, m)
        Centroids matrix (optionally normalized, e.g., in (0,1)).
    ranks : list or tuple of int
        List of ranks [r0, r1, ..., r_{p-1}] for each factor.
    T : int, optional
        Maximum number of coordinate descent iterations.

    Returns
    -------
    B_rec : np.ndarray
        Reconstructed centroids, same shape as B_centroids.
    B_list : list of np.ndarray
        List [B0, B1, ..., B_{p-1}] of factor matrices,
        where Bl has shape (ranks[l], m).
    """
    ranks = list(ranks)
    p = len(ranks)
    total_rows, m = B_centroids.shape
    assert total_rows == int(np.prod(ranks)), \
        "B_centroids rows must equal product of ranks."

    # Initialize factors (you already have makerealdata_v2)
    B_list = [makerealdata_v2(r, m) for r in ranks]

    # thresholds for premature termination
    thresh_up = 1e11   # detect divergence
    thresh_low = 1e-4  # detect convergence

    all_diffs = []
    all_diffs2 = []

    shape_r = tuple(ranks)

    for t in range(T):

        # Block-coordinate updates over each factor
        for l in range(p):
            # Numerator and denominator for closed-form updates
            num = np.zeros((ranks[l], m), dtype=B_centroids.dtype)
            den = np.zeros((ranks[l], m), dtype=B_centroids.dtype)

            # Accumulate over all centroids rows
            for linear_idx in range(total_rows):
                multi_idx = np.unravel_index(linear_idx, shape_r)
                i_l = multi_idx[l]

                # Product of all factors except l, for each feature
                v = np.ones(m, dtype=B_centroids.dtype)
                for d, idx_d in enumerate(multi_idx):
                    if d == l:
                        continue
                    v *= B_list[d][idx_d, :]

                b = B_centroids[linear_idx, :]  # shape (m,)

                num[i_l, :] += b * v
                den[i_l, :] += v ** 2

            # Closed-form update: x = (b^T v) / (v^T v)
            B_list[l][:, :] = num / (den + 1e-8)

        # Monitoring reconstruction error
        if (t % 100 == 0) and (t > 10):
            B_rec = hadamard_cartesian_rows(B_list)
            diff = B_centroids - B_rec
            flag = np.sum(np.abs(diff))
            flag2 = np.sum(diff ** 2)
            all_diffs.append(flag)
            all_diffs2.append(flag2)

            if flag2 > thresh_up or flag2 < thresh_low:
                print("Terminating at iter", t,
                      "- loss =", float(flag2))
                break

    B_rec = hadamard_cartesian_rows(B_list)
    return B_rec, B_list



class PP_KRkmeansP:
    """
    Post-processing-based naive Khatri-Rao clustering with p protocentroid sets.

    Parameters
    ----------
    D : np.ndarray
        Input data matrix, shape (n, m).
    ranks : list or tuple of int
        [r0, r1, ..., r_{p-1}] cardinalities of the protocentroid sets.
        The total number of clusters is prod(ranks).
    standardize : bool, optional
        Standardize the dataset before clustering (default: False).
    n_reps : int, optional
        Number of KMeans initializations (n_init).
    lr : float, optional
        Placeholder (not used here, kept for API compatibility).
    n_epochs : int, optional
        Max iterations for the coordinate descent factorization.
    max_iter : int, optional
        Max iterations for k-means (if you want to pass to KMeans).
    """
    def __init__(self, D, ranks,
                 standardize=False,
                 n_reps=10,
                 lr=0.01,
                 n_epochs=1000,
                 max_iter=50):

        if standardize:
            scaler = StandardScaler()
            D = scaler.fit_transform(D)

        self.D = D.astype('float32')
        self.ranks = list(ranks)
        self.p = len(self.ranks)
        self.n, self.m = D.shape
        self.lr = lr
        self.n_epochs = n_epochs
        self.max_iter = max_iter
        self.n_reps = n_reps

        self.k = int(np.prod(self.ranks))  # total clusters

    def run_k_means(self):
        """Run vanilla k-means with k = prod(ranks)."""
        kmeans = KMeans(
            n_clusters=self.k,
            n_init=self.n_reps,
            init="random",
            random_state=42
            # you can pass max_iter=self.max_iter if desired
        )
        kmeans.fit(self.D)
        self.labels = kmeans.labels_
        self.centroids = kmeans.cluster_centers_
        assert self.centroids.shape == (self.k, self.m)

    def compute_loss_v2(self):
        """Compute inertia (sum of squared distances to reconstructed centroids)."""
        return np.sum((self.estimated_data - self.D) ** 2)

    def decompose(self):
        """
        Decompose centroids into protocentroids via p-factor Hadamard factorization.

        Returns
        -------
        loss : float
            Reconstruction loss on the data.
        result : list
            [B_estimate, B0, B1, ..., B_{p-1}] where:
              - B_estimate is the reconstructed centroids (shape (k, m))
              - Bℓ is the ℓ-th protocentroid matrix (shape (ranks[ℓ], m))
        """
        B_estimate, B_list = coordinate_descent_multi_hadamard(
            self.centroids,
            ranks=self.ranks,
            T=self.n_epochs
        )
        self.estimated_centroids = B_estimate

        # Re-assign points to these reconstructed centroids
        assignments = []
        for x in self.D:
            dists = np.linalg.norm(self.estimated_centroids - x, axis=1)
            assignments.append(np.argmin(dists))

        assignments = np.array(assignments)
        self.labels = assignments
        self.estimated_data = self.estimated_centroids[self.labels]

        l = self.compute_loss_v2()
        return l, [B_estimate] + B_list

    def fit(self):
        """
        Fit: run k-means and decompose centroids into p protocentroid sets.

        Returns
        -------
        loss : float
        Bs : list
            [B_estimate, B0, B1, ..., B_{p-1}]
        labels : np.ndarray
            Final cluster assignments for each sample.
        """
        self.run_k_means()
        loss, Bs = self.decompose()
        return loss, Bs, self.labels




def run_pp_p(X, ranks, n_reps):
    """
    Run naive Khatri-Rao k-means with p protocentroid sets.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix (n, m).
    ranks : list or tuple of int
        [r0, r1, ..., r_{p-1}] cardinalities of protocentroid sets.
    n_reps : int
        Number of repetitions for KMeans initialization (n_init).

    Returns
    -------
    loss : float
    Bs : list
        [B_estimate, B0, B1, ..., B_{p-1}]
    labels : np.ndarray
        Final cluster assignments for each sample.
    """
    pp = PP_KRkmeansP(X, ranks, n_reps=n_reps)
    loss, Bs, labels = pp.fit()
    return loss, Bs, labels


def run_kr_k_means(
    X,
    rlist,
    n_reps=20,
    init_type="random",
    operator="sum",
    impl="original",
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
        Final cluster assignments for each sample (flat index in [0, r1*r2-1]).
    """

    

    # --- construct model ---
    dKM = KrKMeans(
        X,
        rlist,
        standardize=False,
        operator=operator    )

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

    A_list, B_list = bestABs
    return l_lowes_bs, [A_list, B_list], best_assignments









