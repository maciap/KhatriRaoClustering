import numpy as np
import random
from sklearn.preprocessing import StandardScaler
import warnings
from scipy.spatial.distance import cdist
from .kr_k_means_utils import compute_loss  


def compute_centroid_movement_from_semicentroids(B1, B2, B1_prev, B2_prev, operator="product"):
    """
    Compute the squared movement of all full centroids between two iterations.

    This matches:
        np.sum((allcenters - previous_allcenters) ** 2)
    where allcenters enumerates all operator-combined centroids (i, j), without
    materializing the full h1*h2 centroid matrix.
    """
    h1, m = B1.shape
    h2 = B2.shape[0]
    movement = 0.0

    if operator == "product":
        for i in range(h1):
            b1 = B1[i]
            b1p = B1_prev[i]
            for j in range(h2):
                diff = (b1 * B2[j]) - (b1p * B2_prev[j])
                movement += float(np.dot(diff, diff))
    else:  # "sum"
        for i in range(h1):
            b1 = B1[i]
            b1p = B1_prev[i]
            for j in range(h2):
                diff = (b1 + B2[j]) - (b1p + B2_prev[j])
                movement += float(np.dot(diff, diff))

    return movement


class KrKMeans:
    def __init__(self, D, h1, h2, standardize=True, operator="product"):
        """
        Khatri-Rao K-Means with two sets of protocentroids. Space-efficient implementation. 

        Parameters
        ----------
        D : np.ndarray
            Input data of shape (n, m).
        h1 : int
            Number of protocentroids in B1.
        h2 : int
            Number of protocentroids in B2.
        standardize : bool, optional
            Whether to z-score standardize the data.
        operator : {"product", "sum"}
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

        # Stored state after fitting
        self.A_1 = None
        self.A_2 = None
        self.B_1 = None
        self.B_2 = None

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
        dists = cdist(self.D, centroids_arr, metric="sqeuclidean")  # (n, #centroids)
        squared_distances = np.min(dists, axis=1)  # (n,)
        return squared_distances

    def kmeans_plus_plus_init_deterministic(self):
        """
        KMeans++-style initialization using a farthest-point rule.
        """
        warnings.warn("Using ++ initialization")

        # Initial centroid used to seed distance tracking
        centroid_vec = self.D[random.randrange(len(self.D))]
        centroids = [centroid_vec]

        # Initialize B1[0] and B2[0] so their operator-combination equals centroid_vec
        centroid1 = self.D[random.randrange(len(self.D))]
        centroid2 = centroid_vec / centroid1 if self.operator == "product" else centroid_vec - centroid1

        B1 = np.zeros((self.h1, self.m), dtype=self.D.dtype)
        B2 = np.zeros((self.h2, self.m), dtype=self.D.dtype)

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
        """
        warnings.warn("Using ++ initialization")

        # Initial centroid used to seed distance tracking
        centroid_vec = self.D[random.randrange(len(self.D))]
        centroids = [centroid_vec]

        # Initialize B1[0] and B2[0] so their operator-combination equals centroid_vec
        centroid1 = self.D[random.randrange(len(self.D))]
        centroid2 = centroid_vec / centroid1 if self.operator == "product" else centroid_vec - centroid1

        B1 = np.zeros((self.h1, self.m), dtype=self.D.dtype)
        B2 = np.zeros((self.h2, self.m), dtype=self.D.dtype)

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
        """
        B_1 = np.zeros((self.h1, self.m), dtype=self.D.dtype)
        B_2 = np.zeros((self.h2, self.m), dtype=self.D.dtype)

        for i in range(self.h1):
            B_1[i, :] = self.D[np.random.choice(self.n)]

        for j in range(self.h2):
            B_2[j, :] = self.D[np.random.choice(self.n)]

        return B_1, B_2

    def init_ones(self):
        """
        Initialize B1 and B2 with ones.
        """
        return np.ones((self.h1, self.m), dtype=self.D.dtype), np.ones((self.h2, self.m), dtype=self.D.dtype)

    def init_zeros(self):
        """
        Initialize B1 and B2 with zeros.
        """
        return np.zeros((self.h1, self.m), dtype=self.D.dtype), np.zeros((self.h2, self.m), dtype=self.D.dtype)

    def find_cluster_membership(self, D, B1, B2):
        """
        Find cluster membership without storing the full h1*h2 centroid set.

        Returns
        -------
        indices_B1 : np.ndarray
            Protocentroid indices in B1, shape (n,).
        indices_B2 : np.ndarray
            Protocentroid indices in B2, shape (n,).
        indices_best : np.ndarray
            Full-centroid indices in {0, ..., h1*h2-1}, shape (n,).
        """
        D = np.asarray(D, dtype=self.D.dtype)
        n = D.shape[0]

        best_dist = np.full(n, np.inf, dtype=np.float32)
        best_i = np.zeros(n, dtype=np.int64)
        best_j = np.zeros(n, dtype=np.int64)

        if self.operator == "product":
            for i in range(self.h1):
                b1 = B1[i, :]
                for j in range(self.h2):
                    c = b1 * B2[j, :]
                    diff = D - c
                    dist = np.einsum("ij,ij->i", diff, diff)  # squared Euclidean
                    mask = dist < best_dist
                    if np.any(mask):
                        best_dist[mask] = dist[mask]
                        best_i[mask] = i
                        best_j[mask] = j
        else:  # "sum"
            for i in range(self.h1):
                b1 = B1[i, :]
                for j in range(self.h2):
                    c = b1 + B2[j, :]
                    diff = D - c
                    dist = np.einsum("ij,ij->i", diff, diff)
                    mask = dist < best_dist
                    if np.any(mask):
                        best_dist[mask] = dist[mask]
                        best_i[mask] = i
                        best_j[mask] = j

        indices_best = best_i * self.h2 + best_j
        return best_i, best_j, indices_best

    def update_A(self, indices, r):
        """
        Build a one-hot assignment matrix from integer labels.
        """
        A = np.zeros((self.n, r), dtype=self.D.dtype)
        A[np.arange(self.n), indices] = 1
        return A

    def compute_closed_form_solution_product(self, thisD, y):
        """
        Closed-form protocentroid update for the product operator.
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
        """
        diff_mean = np.mean(thisD - y, axis=0)
        return diff_mean.reshape((1, self.D.shape[1]))

    def compute_closed_form_solution(self, thisD, y):
        """
        Dispatch the closed-form protocentroid update by operator.
        """
        if self.operator == "product":
            return self.compute_closed_form_solution_product(thisD, y)
        else:
            return self.compute_closed_form_solution_sum(thisD, y)

    def update_B1(self, indices, other_B_indices, current_other_B):
        """
        Update protocentroids in B1 using closed-form updates.
        """
        B = np.zeros((self.h1, self.m), dtype=self.D.dtype)
        for i in range(self.h1):
            idxs = np.where(indices == i)[0]
            if len(idxs) > 0:
                all_other_indices = other_B_indices[idxs]
                B[i, :] = self.compute_closed_form_solution(
                    self.D[idxs, :], current_other_B[all_other_indices, :]
                )
            else:
                B[i, :] = self.D[np.random.choice(self.n)]
        self.B_1 = B
        return B

    def update_B2(self, indices, other_B_indices, current_other_B):
        """
        Update protocentroids in B2 using closed-form updates.
        """
        B = np.zeros((self.h2, self.m), dtype=self.D.dtype)
        for i in range(self.h2):
            idxs = np.where(indices == i)[0]
            if len(idxs) > 0:
                all_other_indices = other_B_indices[idxs]
                B[i, :] = self.compute_closed_form_solution(
                    self.D[idxs, :], current_other_B[all_other_indices, :]
                )
            else:
                B[i, :] = self.D[np.random.choice(self.n)]
        self.B_2 = B
        return B

    def fit(self, n_iter, th_movement=0.0001, verbose=False, init_type="random"):
        """
        Fit Khatri-Rao K-Means without storing h1*h2 centroids.

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

        for itr in range(n_iter):
            # Save previous protocentroids for movement computation
            B1_prev = B_1.copy()
            B2_prev = B_2.copy()

            # Assignment step without centroid materialization
            indices_B_1, indices_B_2, idxs_all = self.find_cluster_membership(self.D, B_1, B_2)
            A_1 = self.update_A(indices_B_1, self.h1)
            A_2 = self.update_A(indices_B_2, self.h2)

            self.A_1 = A_1
            self.A_2 = A_2
            self.B_1 = B_1
            self.B_2 = B_2

            # Update protocentroids
            B_1 = self.update_B1(indices_B_1, indices_B_2, B_2)
            B_2 = self.update_B2(indices_B_2, indices_B_1, B_1)

            if verbose == 2:
                l = compute_loss(A_1, B_1, A_2, B_2, self.D.astype(np.float64), self.operator)
                print(f"iteration: {itr} \\ loss: {l}")

            # Stop if full-centroid movement is below threshold
            movement_centroids = compute_centroid_movement_from_semicentroids(
                B_1, B_2, B1_prev, B2_prev, operator=self.operator
            )
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
        Predict cluster membership of unseen data without centroid materialization.
        """
        new_data = np.asarray(new_data, dtype=self.D.dtype)
        _, _, indices_best = self.find_cluster_membership(new_data, self.B_1, self.B_2)
        return indices_best.flatten()
