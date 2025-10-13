import copy 
import numpy as np 
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from KathriRaokMeans import kr_k_means
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from clustpy.metrics import unsupervised_clustering_accuracy as acc
import pickle
from collections import defaultdict


def makerealdata(n,c,m,noise=0.0, zeroone=False):
    '''
    Generate a synthetic matrix factorization dataset (D = W @ H).

    Parameters
    ----------
    n : int (data points) 
    c : int  (rank).
    m : int (features or variables).
    noise : float, optional (noise level added to the product matrix - default: 0.0)
    zeroone : bool, optional (if True, round W and H to binary (0 or 1) values default: False)

    Returns
    -------
    W : np.ndarray
    H : np.ndarray
    D : np.ndarray (product) 
    '''

    W = np.random.random_sample(size=(n,c))
    H = np.random.random_sample(size=(c,m))
    if zeroone:
        W = np.round(W)
        H = np.round(H)
    D = W@H
    mnoise = noise*(np.random.random_sample(size=D.shape)-0.5)
    D = D+mnoise

    return W,H,D

def makerealdata_v2(r,m):
    '''
    Generate a random real-valued matrix using a standard normal distribution.

    Parameters
    ----------
    r : int
    m : int

    Returns
    -------
    np.ndarray (random matrix of shape (r, m) with entries drawn from N(0, 1))
    '''
    return np.random.normal(size=(r,m))


def half_kron_rows(mat1, mat2): 
    '''
    Compute a "Khatri-Rao" product between two matrices al

    Each output row is the elementwise product of one row from mat1 and one row from mat2.

    Parameters
    ----------
    mat1 : np.ndarray
    mat2 : np.ndarray

    Returns
    -------
    out : np.ndarray (matrix of shape (r1 * r2, m) where each row is mat1[i, :] * mat2[j, :])
    '''

    out=np.zeros(shape=(mat1.shape[0]*mat2.shape[0], mat1.shape[1]))
    cnt = 0
    for i in range(mat1.shape[0]):
        for j in range(mat2.shape[0]): 
            out[cnt,:] = mat1[i,:] * mat2[j,:]
            cnt+=1     
            
    return out 
            

def create_idx_map(r1,r2): 
    '''
    Create a mapping between 2D index pairs and their flattened linear index.

    Parameters
    ----------
    r1 : int (cardinality of first set of semicentroids)
    r2 : int (cardinality of second set of semicentroids)

    Returns
    -------
    map_idxs : dict (dictionary mapping (i, j) → linear index in range [0, r1 * r2 - 1])
    '''

    map_idxs = dict() 
    cnt=0 
    for i in range(r1): 
        for j in range(r2): 
            map_idxs[(i,j)] = cnt 
            cnt+=1 
            
    return map_idxs


def coordinate_descent_half_kron(B_centroids, r1, r2,   T = 5000): 
    '''  gradient  descent to find B_1 and B_2 for a given Khatri-Rao product B
    
    Parameters:
    ----------
    B_centroids: input matrix (should be normalized in (0,1)) 
    r: rank of each Hadamard factor 
    eta: learning rate 
    T: maximum number of iterations/epochs 
    penalty_indices: set of indices to penalize for not being one! 
    
    Returns: 
    ----------
    B_1, B_2: r x m reconstructions
  
    '''       
    # thresholds for premature termination
    thresh_up = 100000000000 # this only to detecte cases of non-convergence
    thresh_low = 0.0001 # converged 
    all_diffs = [] # L1 norm loss
    all_diffs2 = [] # L2 norm loss
    
    m = B_centroids.shape[1]
    B_1 = makerealdata_v2(r1,m) # randomly initializing one factor (using low rank here- might be not needed) 
    B_2 = makerealdata_v2(r2,m)  # other factor initialized as full rank D / D_1 
   
    map_idx1 = create_idx_map(r1, r2) 

    for t in range(T): 
        
        # update B_1 
        for i in range(r1): 
            for h in range(m):    
                B_1[i,h] = np.sum([ (B_centroids[map_idx1[i,j], h] * B_2[j,h]  ) for j in range(r2)] ) / (np.sum( [ (B_2[j,h])**2 for j in range(r2)] ) + 1e-8)
        
        
        # update B_2 
        for j in range(r2): 
            for h in range(m):    
                B_2[j,h] = np.sum([ (B_centroids[map_idx1[i,j],h] * B_1[i,h]) for i in range(r1)] ) / (np.sum( [ (B_1[i,h])**2 for i in range(r1)] ) + 1e-8)
        
        
        # difference term in the gradient 
        if  t % 100==0 and t > 10:
            B_rec = half_kron_rows(B_1, B_2) 
            diff =  ( B_centroids - B_rec )   
            flag = np.sum(np.abs(diff))
            flag2 = np.sum(diff**2)
            all_diffs.append(flag)
            all_diffs2.append(flag2)
            
            # note in practice we may want to terminate when the gradient change is too small 
            if flag2 > thresh_up or flag2 < thresh_low: 
                print("Terminating before max. number of iterations reached.")
                break 
            
    B_rec = half_kron_rows(B_1, B_2) 
    return B_rec, B_1,  B_2


class PP_KRkmeans(): 
    '''
    Implement post-processing-based naive Khatri-Rao clustering 

    Parameters
    ----------
    D : np.ndarray (input data matrix)
    r1 : int (cardinality first set of protocentroids)
    r2 : int (cardinality first set of protocentroids)
    standardize : bool, optional standardize the dataset before clustering (default: False)
    n_reps : int, optional
    lr : float, optional (learning rate)
    n_epochs : int, optional (epochs for gradient descent) 
    max_iter : int, optional (maximum number of iterations-default: 50) 

    '''
    def __init__(self, D, r1, r2, standardize = False, n_reps = 10, lr = 0.01, n_epochs = 1000, max_iter = 50):
        if standardize: 
            scaler = StandardScaler()
            D = scaler.fit_transform(D)
        self.D = D.astype('float32') 
        self.r1 = r1
        self.r2 = r2 
        self.n, self.m = D.shape
        self.lr = lr 
        self.n_epochs = n_epochs 
        self.max_iter = max_iter 
        self.n_reps = n_reps 
    
    def run_k_means(self): 
        ''' run k means ''' 
        kmeans = KMeans(n_clusters=self.r1 * self.r2, n_init = self.n_reps,  init = "random", random_state=42)
        kmeans.fit(self.D)
        self.labels = kmeans.labels_
        self.centroids = kmeans.cluster_centers_
        assert self.centroids.shape == (self.r1 * self.r2,self.m)
        
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
        B_estimate, B1, B2 = coordinate_descent_half_kron(self.centroids, r1=self.r1, r2= self.r2,  T = self.n_epochs)
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

def run_kr_k_means(X, n_clusters_1, n_clusters_2, n_reps=20, init_type = "random", operator = "sum"):   
    '''
    Run Kr Kmeans clustering  

     Parameters
    ----------
    X : np.ndarray (input data matrix)
    n_clusters1 : int (cardinality first set of protocentroids)
    n_clusters_2 : int (cardinality second set of protocentroids)
    n_reps : int (number of repetitions for initialization) 
    init_type : str (type of initialization) 
    operator : str (aggregator function)

    Returns
    -------
    loss : float
    Bs : list List of protocentroids 
    best_assignments : np.ndarray (final cluster assignments for each sample) 
    '''
          
    dKM = kr_k_means.KrKMeans(X, r1=n_clusters_1, r2=n_clusters_2, standardize = False, operator = operator)
    l_lowes_bs = float("inf")
    for _ in range(n_reps):
        ABs, l, assignments  = dKM.fit(n_iter =  200, th_movement = 0.0001,  verbose = False, init_type =init_type)
        if l < l_lowes_bs:
            l_lowes_bs = l
            bestABs = ABs
            best_assignments = assignments
    A_1, B_1, A_2, B_2 = bestABs 

    return l_lowes_bs, [B_1, B_2] , best_assignments


def run_k_means_baseline(X, n_clusters, n_reps=20): 
    '''
    Run k-kmeans clustering  

     Parameters
    ----------
    X : np.ndarray (input data matrix)
    n_clusters : int (number of centroids)

    Returns
    -------
    loss : float
    centroids : np.ndarray (centroids) 
    labels : np.ndarray (final cluster assignments for each sample) 
    '''

    kmeans = KMeans(n_clusters=n_clusters, n_init =n_reps,  init="random",  random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    estimated_data = estimated_data = centroids[labels]   # check too 
    loss = np.sum(np.power((estimated_data - X),2)) 
    return loss, centroids, labels



def compute_metrics(labels, actual_labels): 
    '''
    Compute clustering performance metrics (ARI, ACC, NMI).

    Parameters
    ----------
    labels : np.ndarray (predicted cluster assignments) 
    actual_labels : np.ndarray (ground-truth labels) 

    Returns
    -------
    metrics : dict (dictionary containing Adjusted Rand Index (ARI), Accuracy (ACC), and Normalized Mutual Information (NMI)) 
    '''

    my_ari = ari(actual_labels, labels)
    my_acc = acc(actual_labels,labels)
    my_nmi = nmi(actual_labels, labels)
    return {"ARI": my_ari, "ACC": my_acc, "NMI": my_nmi}

