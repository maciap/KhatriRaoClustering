import numpy as np 
from sklearn.datasets import (make_blobs, 
                            make_swiss_roll,
                            make_classification)
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.model_selection import StratifiedShuffleSplit
from clustpy.data import (
    load_mnist,
    load_optdigits, 
    load_soybean_large,
    load_stickfigures,
    load_symbols, 
    load_har, 
    load_cmu_faces, 
    load_olivetti_faces
)
import clustbench


def stratified_subsample(X: np.ndarray, L: np.ndarray, n_samples: int = 5000, random_state: int = 42):
    """
    Subsample X, L while keeping label proportions (stratified sampling).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples_total, n_features)
    L : np.ndarray, shape (n_samples_total,)
        Labels
    n_samples : int
        Number of rows to keep
    random_state : int
        Reproducibility

    Returns
    -------
    X_sub : np.ndarray
    L_sub : np.ndarray
    """
    n_total = len(L)
    if n_samples > n_total:
        raise ValueError("n_samples cannot be larger than the total number of samples")

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n_samples, random_state=random_state)
    train_idx, _ = next(sss.split(X, L))
    X_sub = X[train_idx]
    L_sub = L[train_idx]
    return X_sub, L_sub


def closest_factors(n_labels):
    """
    Find two integers whose product is n_labels and that are as close as possible.
    Raise an error if n_labels is prime.
     Parameters
    ----------
    n_labels : int (number of labels)

    Returns
    -------
    a,b tuple of int (closest two factors with a * b = n_labels) 

    """
    # Start from the square root of n_labels and go down
    for i in range(int(n_labels**0.5), 0, -1):
        if n_labels % i == 0:
            a = i
            b = n_labels // i
            if a == 1 and b == n_labels:
                raise ValueError(f"{n_labels} is a prime number — no close factors other than 1 and itself.")
            return (a, b)
    
    # Should never reach here
    raise ValueError(f"Cannot find factors for {n_labels}")

def concatenate_mnist(n_samples:int = 10000, n_digits_position_0: int=10, n_digits_position_1: int=10, random_state: np.random.RandomState=None):

    '''
    Generate double MNIST dataset by concatenating MNIST digits


    Parameters
    ----------
    n_samples : int 
    n_digits_position_0 : int (number of digits in the first position) 
    n_digits_position_1 : int (number of digits in the second position) 
    random_state : int (for reproducibility) 


    Returns
    -------
    X_concat : np.ndarray (data) 
    L_concat : np.ndarray (labels) 
    '''

    random_state = check_random_state(random_state)

    X, L = load_mnist(return_X_y=True)

    n_classes = n_digits_position_0 * n_digits_position_1

    L_concat = np.repeat(np.arange(n_classes), n_samples // n_classes)

    L_concat = np.r_[L_concat, np.arange(n_samples % n_classes)]

    assert L_concat.shape[0] == n_samples

    X_concat = np.zeros((n_samples, X.shape[1] * 2))

    samples_per_cluster = [np.where(L == i)[0] for i in range(10)]

    for i in range(n_samples):

        in_first_cluster = samples_per_cluster[L_concat[i] // 10]

        in_second_cluster = samples_per_cluster[L_concat[i] % 10]

        first = X[random_state.choice(in_first_cluster)]

        second = X[random_state.choice(in_second_cluster)]

        combined = np.c_[first.reshape((28, 28)), second.reshape((28, 28))]

        combined_flat = combined.reshape((-1))

        X_concat[i] = combined_flat

    return X_concat, L_concat


def generate_blobs():
    '''
    Generate Blobs dataset

  
    Returns
    -------
    X : np.ndarray (data) 
    L : np.ndarray (labels) 
    '''


    X, L = make_blobs(n_samples=5000, centers=100, n_features=2, random_state=42)
    return X, L

def generate_swiss_roll():
    '''
    Generate Swiss Roll dataset

  
    Returns
    -------
    X : np.ndarray (data) 
    L : np.ndarray (labels) 
    '''

    X, t = make_swiss_roll(n_samples=5000, noise=0.1, random_state=42)
    L = np.digitize(X[:, 0], bins=np.linspace(X[:, 0].min(), X[:, 0].max(), 100)) - 1
    return X, L

def generate_classification():
    '''
    Generate Classification dataset

  
    Returns
    -------
    X : np.ndarray (data) 
    L : np.ndarray (labels) 
    '''

    X, L = make_classification(
        n_samples=5000,
        n_features=10,
        n_informative=10,
        n_redundant=0,
        n_repeated=0,
        n_classes=100,
        n_clusters_per_class=1,
        random_state=42
    )
    return X, L



def shift_to_zero(arr):
    '''
    Ensure dataset labels start from 0 (for consistency) 
    Parameters 
    -------
    arr: np.ndarray like (labels)

  
    Returns
    -------
    X : np.ndarray (data) 
    L : np.ndarray (labels) 
    '''

    arr = np.array(arr) 
    min_val = arr.min()
    if min_val != 0:
        arr = arr - min_val
    return arr



def compute_inertia(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
    """
    Compute total inertia (WCSS) for clustering results.

    Parameters
    ----------
    X : np.ndarray, shape  (data points). 
    labels : np.ndarray (labels) 
        Cluster assignment for each data point.
    centers : np.ndarray (cluster centers) 
        Cluster centers.

    Returns
    -------
    inertia: float (sum of squared distances of points to their cluster center).
    """
    inertia = 0.0
    for k in range(centers.shape[0]):
        cluster_points = X[labels == k]
        if cluster_points.shape[0] > 0:
            dists = np.linalg.norm(cluster_points - centers[k], axis=1) 
            inertia += np.sum(dists)
    return inertia



def load_data(dataset): 
    '''
    Load dataset 

    Parameters
    ----------
    dataset : string (dataset string). 

    Returns
    -------
    X : np.ndarray (data) 
    L : np.ndarray (labels) 
    '''

    if dataset == "optdigits": 
        X, L = load_optdigits(return_X_y=True)
        
    elif dataset == "mnist": 
        X, L = load_mnist(return_X_y=True) 
        X, L = stratified_subsample(X,L,25000)

    elif dataset == "cmu_faces": 
        X, L = load_cmu_faces(return_X_y=True)
        L = L[:,0] # take the first column with 20 clusters 

    elif dataset == "stickfigures": 
        X,Lp = load_stickfigures(return_X_y=True) 
        L = np.unique(Lp, axis=0, return_inverse=True)[1]

    elif dataset == "har": 
        X, L = load_har(return_X_y=True) 

    elif dataset == "olivetti_faces": 
        X, L = load_olivetti_faces(return_X_y=True) 

    elif dataset == "soybean_large": 
        X, L = load_soybean_large(return_X_y=True) 

    elif dataset == "symbols": 
        X, L = load_symbols(return_X_y=True) 

    elif dataset == "double_mnist": 
        X, L = concatenate_mnist(n_samples=25000)

    elif dataset == "r15":
        data_path = os.path.expanduser("clustering-data-v1")
        battery = "sipu"
        dataset_name = "r15"
        b = clustbench.load_dataset(battery, dataset_name, path=data_path)
        X = b.data
        Lpre = b.labels[0]
        L = np.array([x -1 for x in Lpre]) 

    elif dataset == "chameleon_t7_10k":
        data_path = os.path.expanduser("clustering-data-v1")
        battery = "other"
        dataset_name = "chameleon_t7_10k"
        b = clustbench.load_dataset(battery, dataset_name, path=data_path)
        X = b.data
        Lpre = b.labels[0]
        L = np.array([x -1 for x in Lpre]) 

    elif dataset == "blobs": 
        X, L = generate_blobs() 
    
    elif dataset == "swiss_roll":
        X, L = generate_swiss_roll() 
    
    elif dataset == "classification": 
        X, L = generate_classification() 

    else: 
        print("Wrong input dataset. Dataset must be one of optdigits, mnist, cmu_faces, stickfigures, har, olivetti_faces, symbols, soybean_large, double_mnist, r15, chameleon_t7_10k, blobs, swiss_roll or classification.")  
        sys.exit(0) 

    return X,L 