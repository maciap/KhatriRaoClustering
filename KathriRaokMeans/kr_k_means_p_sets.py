import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.spatial.distance import cdist
from .kr_k_means_utils import compute_centroid_movement  # reuse this


class KrKMeans:
    """
    Generalized Khatri-Rao K-Means with p sets of protocentroids.

    Parameters
    ----------
    D : ndarray, shape (n, m)
        Data matrix.
    h_list : list[int]
        Protocentroid counts for each set, e.g. [h1, h2, ..., rp].
    standardize : bool
        Whether to z-score standardize D.
    operator : {"product", "sum"}
        Elementwise operator used to combine protocentroids.

    Notes
    -----
    For p = 2, this matches the two-set variant.
    """

    def __init__(self, D, h_list, standardize=True, operator="product"):
        if standardize:
            scaler = StandardScaler()
            D = scaler.fit_transform(D)

        self.D = D.astype("float32")
        self.n, self.m = self.D.shape

        self.h_list = list(h_list)        # [h1, h2, ..., rp]
        self.p = len(self.h_list)


        assert operator in ["product", "sum"], (
            f"operator must be 'sum' or 'product' but is {operator}"
        )
        self.operator = operator

        # Stored state after fitting
        self.B_list = None    # list of B_k, each (r_k, m)
        self.A_list = None    # list of A_k, each (n, r_k)
        self.allcenters = None

    def _compute_allcenters(self, B_list):
        """
        Compute all full centroids from the protocentroid matrices.

        B_list[k] has shape (r_k, m). The result has shape:
            (h1 * h2 * ... * rp, m)
        where each row is the operator-combination of one index tuple across
        the p protocentroid sets.
        """
        p = self.p
        m = self.m

        allcenters = None
        for k, (Bk, rk) in enumerate(zip(B_list, self.h_list)):
            # Reshape Bk to broadcast over the other protocentroid-set dimensions
            shape = [1] * (p + 1)  # p protocentroid dims + 1 feature dim
            shape[k] = rk
            shape[-1] = m
            Bk_reshaped = Bk.reshape(shape)

            if allcenters is None:
                allcenters = Bk_reshaped
            else:
                if self.operator == "product":
                    allcenters = allcenters * Bk_reshaped
                else:
                    allcenters = allcenters + Bk_reshaped

        # Flatten protocentroid-set dimensions to (prod(h_list), m)
        return allcenters.reshape(-1, m)

    def _flat_to_multi_indices(self, indices_best):
        """
        Convert flat centroid indices into p indices, one per protocentroid set.

        indices_best : array of shape (n,)
        Returns
        -------
        indices_list : list of length p
            indices_list[k] has shape (n,) and indexes B_list[k].
        """
        indices_best = np.asarray(indices_best, dtype=np.int64)
        n = indices_best.shape[0]
        r_sizes = self.h_list
        p = self.p

        indices_list = [np.empty(n, dtype=np.int64) for _ in range(p)]

        for i in range(n):
            k = indices_best[i]
            # Decode from last dimension to first
            for dim in reversed(range(p)):
                r = r_sizes[dim]
                idx = k % r
                k //= r
                indices_list[dim][i] = idx

        return indices_list

    def random_inizialization_B(self):
        """
        Initialize each protocentroid matrix by sampling rows from D.
        """
        B_list = []
        for rk in self.h_list:
            Bk = np.zeros((rk, self.m), dtype=self.D.dtype)
            for i in range(rk):
                row_index = np.random.choice(self.n)
                Bk[i, :] = self.D[row_index]
            B_list.append(Bk)
        return B_list

    def init_ones(self):
        """
        Initialize all protocentroids with ones.
        """
        return [np.ones((rk, self.m), dtype=self.D.dtype) for rk in self.h_list]

    def init_zeros(self):
        """
        Initialize all protocentroids with zeros.
        """
        return [np.zeros((rk, self.m), dtype=self.D.dtype) for rk in self.h_list]

    def find_cluster_membership(self, D):
        """
        Assign samples to the nearest full centroid and return protocentroid indices.

        Returns
        -------
        indices_list : list of length p
            indices_list[k] is (n,) with protocentroid index for B_list[k].
        indices_best : ndarray, shape (n,)
            Flat centroid index in [0, prod(h_list)-1].
        """
        distances = cdist(D, self.allcenters)
        indices_best = np.argmin(distances, axis=1)
        indices_list = self._flat_to_multi_indices(indices_best)
        return indices_list, indices_best

    def update_A(self, indices, rk):
        """
        Build a one-hot assignment matrix from integer labels.
        """
        A = np.zeros((self.n, rk), dtype=self.D.dtype)
        A[np.arange(self.n), indices] = 1
        return A

    def compute_closed_form_solution_product(self, thisD, y):
        """
        Closed-form protocentroid update for the product operator.

        thisD, y have shape (n_i, m). The return value has shape (1, m).
        """
        n_i, m = thisD.shape
        solutionvec = np.zeros(m, dtype=thisD.dtype)
        for j in range(m):
            den = np.sum(y[:, j] ** 2)
            if den > 0:
                solutionvec[j] = np.sum(thisD[:, j] * y[:, j]) / den
            else:
                solutionvec[j] = 0
        return solutionvec.reshape((1, self.D.shape[1]))

    def compute_closed_form_solution_sum(self, thisD, y):
        """
        Closed-form protocentroid update for the sum operator.

        thisD, y have shape (n_i, m). The return value has shape (1, m).
        """
        n_i, m = thisD.shape
        solutionvec = np.zeros(m, dtype=thisD.dtype)
        for j in range(m):
            solutionvec[j] = np.sum(thisD[:, j] - y[:, j]) / n_i
        return solutionvec.reshape((1, self.D.shape[1]))

    def compute_closed_form_solution(self, thisD, y):
        """
        Dispatch the closed-form protocentroid update by operator.
        """
        if self.operator == "product":
            return self.compute_closed_form_solution_product(thisD, y)
        else:
            return self.compute_closed_form_solution_sum(thisD, y)

    def update_Bk(self, k, indices_list):
        """
        Update protocentroids in B_list[k].

        Parameters
        ----------
        k : int
            Index of the protocentroid set to update.
        indices_list : list of length p
            indices_list[j] contains protocentroid assignments for B_list[j].

        Returns
        -------
        Bk : ndarray, shape (r_k, m)
            Updated protocentroid matrix for B_list[k].
        """
        rk = self.h_list[k]
        Bk_new = np.zeros((rk, self.m), dtype=self.D.dtype)

        for i in range(rk):
            idxs = np.where(indices_list[k] == i)[0]
            if len(idxs) > 0:
                thisD = self.D[idxs, :]

                # Build y from the other protocentroid sets
                if self.operator == "product":
                    y = np.ones_like(thisD)
                else:
                    y = np.zeros_like(thisD)

                for j in range(self.p):
                    if j == k:
                        continue
                    Bj = self.B_list[j]
                    idxs_j = indices_list[j][idxs]
                    part = Bj[idxs_j, :]
                    if self.operator == "product":
                        y *= part
                    else:
                        y += part

                Bk_new[i, :] = self.compute_closed_form_solution(thisD, y)
            else:
                # Reinitialize an unused protocentroid from a random data point
                Bk_new[i, :] = self.D[np.random.choice(self.n)]

        self.B_list[k] = Bk_new
        return Bk_new

    def compute_loss(self, A_list, B_list):
        """
        Reconstruction loss for p protocentroid sets.

        If operator == "product":
            D_hat = (A_1 @ B_1) * (A_2 @ B_2) * ... * (A_p @ B_p)
        If operator == "sum":
            D_hat = (A_1 @ B_1) + (A_2 @ B_2) + ... + (A_p @ B_p)

        Returns
        -------
        float
            Total squared error ||D_hat - D||_F^2.
        """
        n, m = self.D.shape

        if self.operator == "product":
            D_hat = np.ones((n, m), dtype=self.D.dtype)
        else:
            D_hat = np.zeros((n, m), dtype=self.D.dtype)

        for A, B in zip(A_list, B_list):
            AB = A @ B  # shape (n, m)
            if self.operator == "product":
                D_hat *= AB
            else:
                D_hat += AB

        return float(np.sum((D_hat - self.D) ** 2))

    def fit(self, n_iter, th_movement=0.0001, verbose=False, init_type="random"):
        """
        Fit p-set Khatri-Rao K-Means.

        Parameters
        ----------
        n_iter : int
            Maximum number of iterations.
        th_movement : float
            Threshold on full-centroid movement to declare convergence.
        verbose : int
            If 2, prints the loss each iteration.
        init_type : {"random", "ones", "zeros"} or list
            Initialization strategy or explicit list of protocentroid matrices.

        Returns
        -------
        (A_list, B_list), loss, idxs_all
            A_list: list of assignment matrices A_k (n, r_k),
            B_list: list of protocentroid matrices B_k (r_k, m),
            loss: final squared loss,
            idxs_all: flat centroid indices for each data point.
        """
        # Initialize protocentroids
        if isinstance(init_type, list):
            # Expect a list of B_k
            assert len(init_type) == self.p, "init_type as list must have length p"
            self.B_list = init_type
        elif init_type == "random":
            self.B_list = self.random_inizialization_B()
        elif init_type == "ones":
            self.B_list = self.init_ones()
        elif init_type == "zeros":
            self.B_list = self.init_zeros()
        else:
            print(
                "Invalid init_type. Use 'random', 'ones', 'zeros', or a list of B_k. "
                "Defaulting to random."
            )
            self.B_list = self.random_inizialization_B()

        # Initialize full centroids for the first movement computation
        self.allcenters = np.full(
            (np.prod(self.h_list), self.m), np.inf, dtype=self.D.dtype
        )

        for itr in range(n_iter):
            # Update full centroids
            self.previous_allcenters = np.copy(self.allcenters)
            self.allcenters = self._compute_allcenters(self.B_list)

            # Assignment step
            indices_list, idxs_all = self.find_cluster_membership(self.D)
            A_list = [
                self.update_A(indices_list[k], self.h_list[k])
                for k in range(self.p)
            ]

            # Store current assignments
            self.A_list = A_list

            # Update protocentroids in sequence
            for k in range(self.p):
                self.update_Bk(k, indices_list)

            if verbose == 2:
                l = self.compute_loss(self.A_list, self.B_list)
                print(f"iteration: {itr} \\ loss: {l}")

            # Stop if full-centroid movement is below threshold
            movement_centroids = compute_centroid_movement(
                self.allcenters, self.previous_allcenters
            )

            if movement_centroids < th_movement:
                print(f"Converged after {itr + 1} iterations.")
                break

        if itr == n_iter - 1:
            print(f"Maximum number of iterations ({itr + 1}) reached.")

        final_loss = self.compute_loss(self.A_list, self.B_list)
        return (self.A_list, self.B_list), final_loss, idxs_all

    def predict(self, new_data):
        """
        Predict cluster membership of unseen data.

        Returns
        -------
        indices_best : ndarray, shape (n_new,)
            Flat centroid indices for each new point.
        """
        allcenters = self._compute_allcenters(self.B_list)
        distances = cdist(new_data, allcenters)
        indices_best = np.argmin(distances, axis=1)
        return indices_best.flatten()
