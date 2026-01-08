import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import warnings
from .kr_k_means_utils import find_current_centers, compute_loss, compute_centroid_movement
from scipy.spatial.distance import cdist


class KrKMeans:
    def __init__(self, D, h1, h2, standardize=True, operator="product"):
        """
        Khatri-Rao K-Means with two sets of protocentroids. Time-efficient implementation.

        Given data D in R^{n x m}, this method maintains two protocentroid
        matrices B1 (h1 x m) and B2 (h2 x m) and constructs h1*h2 full centroids
        via an aggregator (product or sum).

        Parameters
        ----------
        D : np.ndarray
            Input data of shape (n, m).
        h1 : int
            Number of protocentroids in B1.
        h2 : int
            Number of protocentroids in B2.
        standardize : bool, optional
            If True, standardize each feature of D to zero mean and unit variance.
        operator : {"product", "sum"}, optional
            Elementwise operator used to form full centroids from B1 and B2.
        """
        if standardize:
            scaler = StandardScaler()
            D = scaler.fit_transform(D)

        self.D = D.astype("float32")
        self.h1 = h1
        self.h2 = h2
        self.n, self.m = self.D.shape

        assert operator in ["product", "sum"], (
            f"operator must be 'sum' or 'product' but is {operator}"
        )
        self.operator = operator

  
    def _compute_kpp_distances(self, centroids):
        """
        Compute squared distances to the nearest selected centroid.

        Parameters
        ----------
        centroids : list of np.ndarray
            List of centroid vectors with shape (m,).

        Returns
        -------
        squared_distances : np.ndarray
            Array of shape (n,) with the minimum squared distance for each sample.
        """
        centroids_arr = np.vstack(centroids)  # (#centroids, m)
        # Squared Euclidean distances from samples to selected centroids
        dists = cdist(self.D, centroids_arr, metric="sqeuclidean")  # (n, #centroids)
        squared_distances = np.min(dists, axis=1)  # (n,)
        return squared_distances

    def kmeans_plus_plus_init_deterministic(self):
        """
        KMeans++-style initialization using a farthest-point rule.

        Returns
        -------
        B1 : np.ndarray
            Initial protocentroids for B1, shape (h1, m).
        B2 : np.ndarray
            Initial protocentroids for B2, shape (h2, m).

        """
        warnings.warn("Using ++ initialization")

        # Initial centroid used to seed distance tracking
        centroid_vec = self.D[random.randrange(len(self.D))]
        centroids = [centroid_vec]

        # Initialize B1[0] and B2[0] so their operator-combination equals centroid_vec
        centroid1 = self.D[random.randrange(len(self.D))]
        centroid2 = centroid_vec / centroid1 if self.operator == "product" else centroid_vec - centroid1

        B1 = np.zeros((self.h1, self.m))
        B2 = np.zeros((self.h2, self.m))

        B1[0, :] = centroid1
        B2[0, :] = centroid2

        # Fill missing protocentroids using farthest-point selections
        for c1 in range(self.h1):
            for c2 in range(self.h2):
                # Squared distances to the nearest selected centroid
                squared_distances = self._compute_kpp_distances(centroids)

                # Select the farthest data point
                centroid_idx = np.argmax(squared_distances)

                # Set missing rows in B1 and/or B2 to match the selected point under the operator
                if np.sum(B1[c1, :]) == 0 and np.sum(B2[c2, :]) == 0:
                    centroid1 = self.D[random.randrange(len(self.D))]
                    if self.operator == "product":
                        centroid2 = self.D[centroid_idx] / centroid1
                    else:
                        centroid2 = self.D[centroid_idx] - centroid1
                    B1[c1, :] = centroid1
                    B2[c2, :] = centroid2
                elif np.sum(B1[c1, :]) != 0 and np.sum(B2[c2, :]) == 0:
                    if self.operator == "product":
                        B2[c2, :] = self.D[centroid_idx] / B1[c1, :]
                    else:
                        B2[c2, :] = self.D[centroid_idx] - B1[c1, :]
                elif np.sum(B1[c1, :]) == 0 and np.sum(B2[c2, :]) != 0:
                    if self.operator == "product":
                        B1[c1, :] = self.D[centroid_idx] / B2[c2, :]
                    else:
                        B1[c1, :] = self.D[centroid_idx] - B2[c2, :]

                # Add the selected point to the set used for distance tracking
                centroids.append(self.D[centroid_idx])

                # Stop once B1 and B2 have no all-zero rows
                if not np.any(np.all(B1 == 0, axis=1)) and not np.any(np.all(B2 == 0, axis=1)):
                    break

        return B1, B2

    def kmeans_plus_plus_init_random(self):
        """
        KMeans++-style initialization using sampling by squared distance.

        Returns
        -------
        B1 : np.ndarray
            Initial protocentroids for B1, shape (h1, m).
        B2 : np.ndarray
            Initial protocentroids for B2, shape (h2, m).


        """
        warnings.warn("Using ++ initialization")

        # Initial centroid used to seed distance tracking
        centroid_vec = self.D[random.randrange(len(self.D))]
        centroids = [centroid_vec]

        # Initialize B1[0] and B2[0] so their operator-combination equals centroid_vec
        centroid1 = self.D[random.randrange(len(self.D))]
        centroid2 = centroid_vec / centroid1 if self.operator == "product" else centroid_vec - centroid1

        B1 = np.zeros((self.h1, self.m))
        B2 = np.zeros((self.h2, self.m))

        B1[0, :] = centroid1
        B2[0, :] = centroid2

        # Fill missing protocentroids using KMeans++ sampling
        for c1 in range(self.h1):
            for c2 in range(self.h2):
                # Squared distances to the nearest selected centroid
                squared_distances = self._compute_kpp_distances(centroids)

                # Convert squared distances into sampling probabilities
                total = squared_distances.sum()
                # If all distances are zero, fall back to uniform sampling
                if total == 0:
                    proba = np.full_like(squared_distances, 1.0 / len(squared_distances))
                else:
                    proba = squared_distances / total

                # Sample the next centroid index under the KMeans++ distribution
                centroid_idx = np.random.choice(len(self.D), size=1, p=proba)[0]

                # Set missing rows in B1 and/or B2 to match the selected point under the operator
                if np.sum(B1[c1, :]) == 0 and np.sum(B2[c2, :]) == 0:
                    centroid1 = self.D[random.randrange(len(self.D))]
                    if self.operator == "product":
                        centroid2 = self.D[centroid_idx] / centroid1
                    else:
                        centroid2 = self.D[centroid_idx] - centroid1
                    B1[c1, :] = centroid1
                    B2[c2, :] = centroid2
                elif np.sum(B1[c1, :]) != 0 and np.sum(B2[c2, :]) == 0:
                    if self.operator == "product":
                        B2[c2, :] = self.D[centroid_idx] / B1[c1, :]
                    else:
                        B2[c2, :] = self.D[centroid_idx] - B1[c1, :]
                elif np.sum(B1[c1, :]) == 0 and np.sum(B2[c2, :]) != 0:
                    if self.operator == "product":
                        B1[c1, :] = self.D[centroid_idx] / B2[c2, :]
                    else:
                        B1[c1, :] = self.D[centroid_idx] - B2[c2, :]

                # Add the selected point to the set used for subsequent sampling
                centroids.append(self.D[centroid_idx])

                # Stop once B1 and B2 have no all-zero rows
                if not np.any(np.all(B1 == 0, axis=1)) and not np.any(np.all(B2 == 0, axis=1)):
                    break

        return B1, B2

    def random_inizialization_B(self):
        """
        Initialize B1 and B2 by sampling data points with replacement.

        Returns
        -------
        B1 : np.ndarray
            Initial protocentroids for B1, shape (h1, m).
        B2 : np.ndarray
            Initial protocentroids for B2, shape (h2, m).
        """
        B_1 = np.zeros((self.h1, self.m))
        B_2 = np.zeros((self.h2, self.m))

        for i in range(self.h1):
            row_index = np.random.choice(self.n)
            B_1[i, :] = self.D[row_index]

        for j in range(self.h2):
            row_index = np.random.choice(self.n)
            B_2[j, :] = self.D[row_index]

        return B_1, B_2

    def init_ones(self):
        """
        Initialize B1 and B2 with ones.

        Returns
        -------
        B1 : np.ndarray
            Ones array of shape (h1, m).
        B2 : np.ndarray
            Ones array of shape (h2, m).
        """
        B_1 = np.ones((self.h1, self.m))
        B_2 = np.ones((self.h2, self.m))
        return B_1, B_2

    def init_zeros(self):
        """
        Initialize B1 and B2 with zeros.

        Returns
        -------
        B1 : np.ndarray
            Zeros array of shape (h1, m).
        B2 : np.ndarray
            Zeros array of shape (h2, m).
        """
        B_1 = np.zeros((self.h1, self.m))
        B_2 = np.zeros((self.h2, self.m))
        return B_1, B_2

    def find_cluster_membership(self, D):
        """
        Assign each sample to the nearest full centroid.

        Parameters
        ----------
        D : np.ndarray
            Data matrix of shape (n, m).

        Returns
        -------
        indices_B1 : np.ndarray
            Protocentroid indices in B1, shape (n,).
        indices_B2 : np.ndarray
            Protocentroid indices in B2, shape (n,).
        indices_best : np.ndarray
            Full-centroid indices in {0, ..., h1*h2-1}, shape (n,).
        """
        distances = cdist(D, self.allcenters)  # (n, h1*h2)
        indices_best = np.argmin(distances, axis=1)
        return (indices_best // self.h2).flatten(), (indices_best % self.h2).flatten(), indices_best

    def update_A(self, indices, r):
        """
        Build a one-hot assignment matrix from integer labels.

        Parameters
        ----------
        indices : np.ndarray
            Labels of shape (n,).
        r : int
            Number of columns in the one-hot matrix.

        Returns
        -------
        A : np.ndarray
            One-hot matrix of shape (n, r).
        """
        A = np.zeros((self.n, r))
        A[np.arange(self.n), indices] = 1
        return A


    def compute_closed_form_solution_product(self, thisD, y):
        """
        Closed-form protocentroid update for the product operator.

        Parameters
        ----------
        thisD : np.ndarray
            Assigned samples, shape (n_i, m).
        y : np.ndarray
            Aligned counterpart values, shape (n_i, m).

        Returns
        -------
        b : np.ndarray
            Updated protocentroid row of shape (1, m).
        """
        num = np.einsum("ij,ij->j", thisD, y)
        den = np.sum(y ** 2, axis=0)

        solutionvec = np.zeros_like(num)
        mask = den > 0
        solutionvec[mask] = num[mask] / den[mask]
        return solutionvec.reshape((1, self.D.shape[1]))

    def compute_closed_form_solution_sum(self, thisD, y):
        """
        Closed-form protocentroid update for the sum operator.

        Parameters
        ----------
        thisD : np.ndarray
            Assigned samples, shape (n_i, m).
        y : np.ndarray
            Aligned counterpart values, shape (n_i, m).

        Returns
        -------
        b : np.ndarray
            Updated protocentroid row of shape (1, m).
        """
        diff_mean = np.mean(thisD - y, axis=0)
        return diff_mean.reshape((1, self.D.shape[1]))

    def compute_closed_form_solution(self, thisD, y):
        """
        Dispatch the closed-form protocentroid update by operator.

        Parameters
        ----------
        thisD : np.ndarray
            Assigned samples, shape (n_i, m).
        y : np.ndarray
            Aligned counterpart values, shape (n_i, m).

        Returns
        -------
        b : np.ndarray
            Updated protocentroid row of shape (1, m).
        """
        if self.operator == "product":
            return self.compute_closed_form_solution_product(thisD, y)
        else:
            return self.compute_closed_form_solution_sum(thisD, y)


    def update_B1(self, indices, other_B_indices, current_other_B):
        """
        Update protocentroids in B1 using closed-form updates.

        Parameters
        ----------
        indices : np.ndarray
            Protocentroid assignments for B1, shape (n,).
        other_B_indices : np.ndarray
            Protocentroid assignments for B2, shape (n,).
        current_other_B : np.ndarray
            Current B2 matrix, shape (h2, m).

        Returns
        -------
        B1 : np.ndarray
            Updated B1 matrix, shape (h1, m).
        """
        B = np.zeros((self.h1, self.m))
        for i in range(self.h1):
            idxs = np.where(indices == i)[0]
            if len(idxs) > 0:
                all_other_indices = other_B_indices[idxs]
                B[i, :] = self.compute_closed_form_solution(
                    self.D[idxs, :], current_other_B[all_other_indices, :]
                )
            else:
                # Reinitialize an empty protocentroid from a random data point
                B[i, :] = self.D[np.random.choice(self.n)]

        self.B_1 = B
        # Loss is not computed in this method
        return B

    def update_B2(self, indices, other_B_indices, current_other_B):
        """
        Update protocentroids in B2 using closed-form updates.

        Parameters
        ----------
        indices : np.ndarray
            Protocentroid assignments for B2, shape (n,).
        other_B_indices : np.ndarray
            Protocentroid assignments for B1, shape (n,).
        current_other_B : np.ndarray
            Current B1 matrix, shape (h1, m).

        Returns
        -------
        B2 : np.ndarray
            Updated B2 matrix, shape (h2, m).
        """
        B = np.zeros((self.h2, self.m))
        for i in range(self.h2):
            idxs = np.where(indices == i)[0]
            if len(idxs) > 0:
                all_other_indices = other_B_indices[idxs]
                B[i, :] = self.compute_closed_form_solution(
                    self.D[idxs, :], current_other_B[all_other_indices, :]
                )
            else:
                # Reinitialize an empty protocentroid from a random data point
                B[i, :] = self.D[np.random.choice(self.n)]

        self.B_2 = B
        # Loss is not computed in this method
        return B


    def fit(self, n_iter, th_movement=0.0001, verbose=False, init_type="random"):
        """
        Fit Khatri-Rao K-Means.

        Parameters
        ----------
        n_iter : int
            Maximum number of iterations.
        th_movement : float, optional
            Convergence threshold on full-centroid movement.
        verbose : bool or int, optional
            If set to 2, prints iteration-wise loss values.
        init_type : {"kmeans++ random", "kmeans++ deterministic", "random", "ones"} or list
            Initialization strategy, or [B1_init, B2_init].

        Returns
        -------
        params : list
            [A_1, B_1, A_2, B_2].
        l : float
            Final objective value.
        idxs_all : np.ndarray
            Full-centroid assignments of shape (n,).
        """
        # Initialize protocentroids
        if init_type == "kmeans++ random":
            B_1, B_2 = self.kmeans_plus_plus_init_random()
        elif init_type == "kmeans++ deterministic":
            B_1, B_2 = self.kmeans_plus_plus_init_deterministic()
        elif init_type == "random":
            B_1, B_2 = self.random_inizialization_B()
        elif init_type == "ones":
            B_1, B_2 = self.init_ones()
        elif isinstance(init_type, list):
            B_1, B_2 = init_type
        else:
            print(
                "Invalid input. Please input a list of semi-centroids or enter a "
                "valid string specifying the initialization type "
                "(kmeans++ random / kmeans++ deterministic / random). Defaulting to random."
            )
            B_1, B_2 = self.random_inizialization_B()

        # Initialize full-centroid array
        self.allcenters = np.full((self.h1 * self.h2, self.m), np.inf, dtype="float32")

        for itr in range(n_iter):
            # Construct full centroids from protocentroids
            self.previous_allcenters = np.copy(self.allcenters)
            all_centers = find_current_centers(B_1, B_2, self.h1, self.h2, self.m, self.operator)
            self.allcenters = all_centers

            # Assign samples and build one-hot assignment matrices
            indices_B_1, indices_B_2, idxs_all = self.find_cluster_membership(self.D)
            A_1 = self.update_A(indices_B_1, self.h1)
            A_2 = self.update_A(indices_B_2, self.h2)

            # Store current state
            self.A_1 = A_1
            self.A_2 = A_2
            self.B_1 = B_1
            self.B_2 = B_2

            # Update protocentroids
            B_1 = self.update_B1(indices_B_1, indices_B_2, B_2)
            B_2 = self.update_B2(indices_B_2, indices_B_1, B_1)

            # Print loss if requested
            if verbose == 2:
                l = compute_loss(A_1, B_1, A_2, B_2, self.D, self.operator)
                print(f"iteration: {itr} \\ loss: {l}")

            # Stop if full-centroid movement is below threshold.
            movement_centroids = compute_centroid_movement(self.allcenters, self.previous_allcenters)
            if movement_centroids < th_movement:
                print(f"Converged after {itr + 1} iterations.")
                break

        if itr == n_iter - 1:
            print(f"Maximum number of iterations ({itr + 1}) reached.")

        # Compute final objective value
        l = compute_loss(A_1, B_1, A_2, B_2, self.D, self.operator)
        return [A_1, B_1, A_2, B_2], l, idxs_all

   
    def predict(self, new_data):
        """
        Assign new samples to the nearest fitted full centroids.

        Parameters
        ----------
        new_data : np.ndarray
            Data matrix of shape (n_new, m).

        Returns
        -------
        labels : np.ndarray
            Full-centroid indices of shape (n_new,).
        """
        _, _, indices_best = self.find_cluster_membership(new_data)
        return indices_best.flatten()
