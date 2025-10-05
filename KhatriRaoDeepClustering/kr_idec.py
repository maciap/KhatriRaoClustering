import numpy as np
import torch
from clustpy.deep._utils import encode_batchwise, predict_batchwise, mean_squared_error
from clustpy.deep._train_utils import get_default_deep_clustering_initialization
from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
from sklearn.cluster import KMeans
from sklearn.base import ClusterMixin
from collections.abc import Callable
from clustpy.deep.dec import _DEC_Module, _dec_predict, _dec_compression_loss_fn, _dec_compression_value
import os 
import sys 
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dkm_path = os.path.join(parent_dir, 'KathriRaokMeans')
sys.path.append(dkm_path)
from kr_k_means import KrKMeans
from clustpy.utils import plot_with_transformation
import tqdm


def _kr_dec(X: np.ndarray, n_clusters_1: int, n_clusters_2: int, alpha: float, batch_size: int, pretrain_optimizer_params: dict,
         clustering_optimizer_params: dict, pretrain_epochs: int, clustering_epochs: int,
         optimizer_class: torch.optim.Optimizer, ssl_loss_fn: Callable | torch.nn.modules.loss._Loss,
         neural_network: torch.nn.Module | tuple, neural_network_weights: str, embedding_size: int,
         clustering_loss_weight: float, ssl_loss_weight: float, custom_dataloaders: tuple,
         augmentation_invariance: bool, initial_clustering_class: ClusterMixin, initial_clustering_params: dict,
         device: torch.device, random_state: np.random.RandomState) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module):
    """
    Start the Khatri-Rao DEC clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray / torch.Tensor
        the given data set. Can be a np.ndarray or a torch.Tensor
    n_clusters_1 : int
        number of protocentroids in set 1
    n_clusters_2 : int
        number of protocentroids in set 2
    alpha : float
        alpha value for the prediction
    batch_size : int
        size of the data batches
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network
    clustering_epochs : int
        number of epochs for the actual clustering procedure
    optimizer_class : torch.optim.Optimizer
        the optimizer
    ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
    neural_network : torch.nn.Module | tuple
        the input neural network.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network.
    embedding_size : int
        size of the embedding within the neural network
    clustering_loss_weight : float
        weight of the clustering loss
    ssl_loss_weight : float
        weight of the self-supervised learning (ssl) loss
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        Can also be a tuple of strings, where the first entry is the path to a saved trainloader and the second entry the path to a saved testloader.
        In this case the dataloaders will be loaded by torch.load(PATH).
        If None, the default dataloaders will be used
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn 
        cluster assignments that are invariant to the augmentation transformations
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining
    initial_clustering_params : dict
        parameters for the initial clustering class
    device : torch.device
        The device on which to perform the computations
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module)
        The labels as identified by a final KMeans execution,
        The cluster centers as identified by a final KMeans execution,
        The labels as identified by DEC after the training terminated,
        The cluster centers as identified by DEC after the training terminated,
        The final neural network
    """

    n_clusters = n_clusters_1 * n_clusters_2
    # Get initial setting (device, dataloaders, pretrained AE and initial clustering result)
    device, trainloader, testloader, _, neural_network, embedded_data_init, _, _, _, _ = get_default_deep_clustering_initialization(
        X, n_clusters, batch_size, pretrain_optimizer_params, pretrain_epochs, optimizer_class, ssl_loss_fn,
        neural_network, embedding_size, custom_dataloaders, None, initial_clustering_params, device,
        random_state, neural_network_weights=neural_network_weights)
    # Setup - use KR k-means for initialization of protocentroids
    operator = "sum"
    n_init = 30
    best_score = np.inf
    best_krkmeans = None
    for init_iter in range(n_init):
        kr_kmeans = KrKMeans(embedded_data_init, n_clusters_1, n_clusters_2, operator=operator)
        _, dkm_loss, _ = kr_kmeans.fit(1000, init_type="random", th_movement=0.0001)
        if dkm_loss < best_score:
            best_score = dkm_loss
            best_krkmeans = kr_kmeans
    init_labels = best_krkmeans.predict(embedded_data_init)
    dec_module = _KR_DEC_Module(best_krkmeans.B_1, best_krkmeans.B_2, operator, alpha, augmentation_invariance).to(device)
    optimizer = optimizer_class(list(neural_network.parameters()), **clustering_optimizer_params)
    optimizer_centroids = optimizer_class(list(dec_module.parameters()),
                                **{"lr":clustering_optimizer_params["lr"]*10})
    # DEC Training loop
    dec_module.fit(neural_network, trainloader, clustering_epochs, device, optimizer, ssl_loss_fn,
                   clustering_loss_weight, ssl_loss_weight, optimizer_centroids)
    # Get labels
    dec_labels = predict_batchwise(testloader, neural_network, dec_module)
    dec_centers = dec_module._get_centers().detach().cpu().numpy()
    # Do reclustering with Kmeans
    embedded_data = encode_batchwise(testloader, neural_network)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embedded_data)
    return kmeans.labels_, kmeans.cluster_centers_, dec_labels, dec_centers, neural_network, dec_module.protocentroids_1.detach().cpu().numpy(), dec_module.protocentroids_2.detach().cpu().numpy()


class _KR_DEC_Module(_DEC_Module):
    """
    The _KR_DEC_Module. Contains most of the algorithm specific procedures like the loss and prediction functions.

    Parameters
    ----------
    init_centers : np.ndarray
        The initial cluster centers
    alpha : double
        alpha value for the prediction method
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)

    Attributes
    ----------
    alpha : float
        the alpha value
    centers : torch.Tensor:
        the cluster centers
    augmentation_invariance : bool
        Is augmentation invariance used
    """

    def __init__(self, protocentroids_1, protocentroids_2, operator: str, alpha: float, augmentation_invariance: bool = False):
        super().__init__(np.array(0.), alpha, augmentation_invariance)
        assert operator in ["product", "sum"], f"operator must be 'sum' or 'product' but is {operator}"
        self.operator = operator
        # Centers are learnable parameters
        self.protocentroids_1 = torch.nn.Parameter(torch.tensor(protocentroids_1), requires_grad=True)
        self.protocentroids_2 = torch.nn.Parameter(torch.tensor(protocentroids_2), requires_grad=True)

    def _get_centers(self):
        dim = self.protocentroids_1.shape[1]
        reshaped_B_1 = self.protocentroids_1.view(self.protocentroids_1.shape[0], 1, dim)
        reshaped_B_2 = self.protocentroids_2.view(1, self.protocentroids_2.shape[0], dim)
        # Element-wise multiplication and reshaping
        if self.operator == "product":
            allcenters = (reshaped_B_1 * reshaped_B_2).view(self.protocentroids_1.shape[0] * self.protocentroids_2.shape[0], dim)       
        else:
            allcenters = (reshaped_B_1 + reshaped_B_2).view(self.protocentroids_1.shape[0] * self.protocentroids_2.shape[0], dim)    
        return allcenters

    def predict(self, embedded: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        Soft prediction of given embedded samples. Returns the corresponding soft labels.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        weights : torch.Tensor
            feature weights for the squared Euclidean distance within the dec_predict method (default: None)

        Returns
        -------
        pred : torch.Tensor
            The predicted soft labels
        """
        centers = self._get_centers()
        pred = _dec_predict(centers, embedded, self.alpha, weights=weights)
        return pred

    def dec_loss(self, embedded: torch.Tensor, weights: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate the DEC loss of given embedded samples.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        weights : torch.Tensor
            feature weights for the squared Euclidean distance within the dec_predict method (default: None)

        Returns
        -------
        loss : torch.Tensor
            the final DEC loss
        """

        centers = self._get_centers()
        prediction = _dec_predict(centers, embedded, self.alpha, weights=weights)
        loss = _dec_compression_loss_fn(prediction)
        return loss

    def dec_augmentation_invariance_loss(self, embedded: torch.Tensor, embedded_aug: torch.Tensor,
                                         weights: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate the DEC loss of given embedded samples with augmentation invariance.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        embedded_aug : torch.Tensor
            the embedded augmented samples
        weights : torch.Tensor
            feature weights for the squared Euclidean distance within the dec_predict method (default: None)

        Returns
        -------
        loss : torch.Tensor
            the final DEC loss
        """
        centers = self._get_centers()

        prediction = _dec_predict(centers, embedded, self.alpha, weights=weights)
        # Predict pseudo cluster labels with clean samples
        clean_target_p = _dec_compression_value(prediction).detach().data
        # Calculate loss from clean prediction and clean targets
        clean_loss = _dec_compression_loss_fn(prediction, clean_target_p)

        # Predict pseudo cluster labels with augmented samples
        aug_prediction = _dec_predict(centers, embedded_aug, self.alpha, weights=weights)

        # Calculate loss from augmented prediction and reused clean targets to enforce that the cluster assignment is invariant against augmentations
        aug_loss = _dec_compression_loss_fn(aug_prediction, clean_target_p)

        # average losses
        loss = (clean_loss + aug_loss) / 2
        return loss

    def fit(self, neural_network: torch.nn.Module, trainloader: torch.utils.data.DataLoader, n_epochs: int,
            device: torch.device, optimizer: torch.optim.Optimizer, ssl_loss_fn: Callable | torch.nn.modules.loss._Loss,
            clustering_loss_weight: float, ssl_loss_weight: float, optimizer_centroids) -> '_DEC_Module':
        """
        Trains the _KR_DEC_Module in place.

        Parameters
        ----------
        neural_network : torch.nn.Module
            the neural network
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        n_epochs : int
            number of epochs for the clustering procedure
        device : torch.device
            device to be trained on
        optimizer : torch.optim.Optimizer
            the optimizer for training
        ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
            self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders
        clustering_loss_weight : float
            weight of the clustering loss
        ssl_loss_weight : float
            weight of the self-supervised learning (ssl) loss

        Returns
        -------
        self : _KR_DEC_Module
            this instance of the _KR_DEC_Module
        """
        tbar = tqdm.trange(n_epochs, desc="DEC training")
        for _ in tbar:
            total_loss = 0
            for batch in trainloader:
                loss = self._loss(batch, neural_network, clustering_loss_weight, ssl_loss_weight, ssl_loss_fn,
                                  device)
                total_loss += loss.item()
                # Backward pass
                optimizer.zero_grad()
                optimizer_centroids.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_centroids.step()
            postfix_str = {"Loss": total_loss}
            tbar.set_postfix(postfix_str)
        return self


class KR_IDEC(_AbstractDeepClusteringAlgo):
    """
    A Kathri Rao variant of IDEC.
    First, a neural_network will be trained (will be skipped if input neural network is given).
    Afterward, KMeans identifies the initial clusters.
    Last, the network will be optimized using the DEC loss function.

    Parameters
    ----------
    n_clusters_1 : int
        number of protocentroids in set 1 
    n_clusters_2 : int
        number of protocentroids in set 2
    alpha : float
        alpha value for the prediction (default: 1.0)
    batch_size : int
        size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate. If None, it will be set to {"lr": 1e-3} (default: None)
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate. If None, it will be set to {"lr": 1e-4} (default: None)
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network (default: 100)
    clustering_epochs : int
        number of epochs for the actual clustering procedure (default: 150)
    optimizer_class : torch.optim.Optimizer
        the optimizer class (default: torch.optim.Adam)
    ssl_loss_fn : Callable | torch.nn.modules.loss._Loss
         self-supervised learning (ssl) loss function for training the network, e.g. reconstruction loss for autoencoders (default: mean_squared_error)
    neural_network : torch.nn.Module | tuple
        the input neural network. If None, a new FeedforwardAutoencoder will be created.
        Can also be a tuple consisting of the neural network class (torch.nn.Module) and the initialization parameters (dict) (default: None)
    neural_network_weights : str
        Path to a file containing the state_dict of the neural_network (default: None)
    embedding_size : int
        size of the embedding within the neural network (default: 10)
    clustering_loss_weight : float
        weight of the clustering loss compared to the reconstruction loss (default: 0.1)
    ssl_loss_weight : float
        weight of the self-supervised learning (ssl) loss (default: 1.0)
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        Can also be a tuple of strings, where the first entry is the path to a saved trainloader and the second entry the path to a saved testloader.
        In this case the dataloaders will be loaded by torch.load(PATH).
        If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn 
        cluster assignments that are invariant to the augmentation transformations (default: False)
    initial_clustering_class : ClusterMixin
        clustering class to obtain the initial cluster labels after the pretraining (default: KMeans)
    initial_clustering_params : dict
        parameters for the initial clustering class. If None, it will be set to {} (default: None)
    device : torch.device
        The device on which to perform the computations.
        If device is None then it will be automatically chosen: if a gpu is available the gpu with the highest amount of free memory will be chosen (default: None)
    random_state : np.random.RandomState | int
        use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

    Attributes
    ----------
    labels_ : np.ndarray
        The final labels (obtained by a final KMeans execution)
    cluster_centers_ : np.ndarray
        The final cluster centers (obtained by a final KMeans execution)
    dec_labels_ : np.ndarray
        The final DEC labels
    dec_cluster_centers_ : np.ndarray
        The final DEC cluster centers
    neural_network_trained_ : torch.nn.Module
        The final neural network
    n_features_in_ : int
        the number of features used for the fitting

  

    References
    ----------
    Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised deep embedding for clustering analysis."
    International conference on machine learning. 2016.
    """

    def __init__(self, n_clusters_1: int = 8, n_clusters_2: int = None, alpha: float = 1.0, batch_size: int = 256,
                 pretrain_optimizer_params: dict = None, clustering_optimizer_params: dict = None,
                 pretrain_epochs: int = 100, clustering_epochs: int = 150,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 ssl_loss_fn: Callable | torch.nn.modules.loss._Loss = mean_squared_error,
                 neural_network: torch.nn.Module | tuple = None, neural_network_weights: str = None,
                 embedding_size: int = 10, clustering_loss_weight: float = 0.1, ssl_loss_weight: float = 1.0, custom_dataloaders: tuple = None,
                 augmentation_invariance: bool = False, initial_clustering_class: ClusterMixin = KMeans,
                 initial_clustering_params: dict = None, device: torch.device = None,
                 random_state: np.random.RandomState | int = None):
        
        super().__init__(batch_size, neural_network, neural_network_weights, embedding_size, device, random_state)
        self.n_clusters_1 = n_clusters_1
        self.n_clusters_2 = n_clusters_2
        self.alpha = alpha
        self.pretrain_optimizer_params = pretrain_optimizer_params
        self.clustering_optimizer_params = clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.ssl_loss_fn = ssl_loss_fn
        self.clustering_loss_weight = clustering_loss_weight
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.initial_clustering_class = initial_clustering_class
        self.initial_clustering_params = initial_clustering_params
        self.ssl_loss_weight = ssl_loss_weight

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'KR_IDEC':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set
        y : np.ndarray
            the labels (can be ignored)

        Returns
        -------
        self : KR_IDEC
            this instance of the DEC algorithm
        """
        if self.n_clusters_2 is None:
            assert np.sqrt(self.n_clusters_1) % 1 == 0
            n_clusters_1 = int(np.sqrt(self.n_clusters_1))
            n_clusters_2 = n_clusters_1
        else:
            n_clusters_1 = self.n_clusters_1
            n_clusters_2 = self.n_clusters_2
        
        ssl_loss_weight = self.ssl_loss_weight if hasattr(self, "ssl_loss_weight") else 0 # DEC does not use ssl loss when clustering
        X, _, random_state, pretrain_optimizer_params, clustering_optimizer_params, initial_clustering_params = self._check_parameters(X, y=y)
        kmeans_labels, kmeans_centers, dec_labels, dec_centers, neural_network, protocentroids_1, protocentroids_2 = _kr_dec(X, n_clusters_1, n_clusters_2, self.alpha,
                                                                                      self.batch_size,
                                                                                      pretrain_optimizer_params,
                                                                                      clustering_optimizer_params,
                                                                                      self.pretrain_epochs,
                                                                                      self.clustering_epochs,
                                                                                      self.optimizer_class,
                                                                                      self.ssl_loss_fn,
                                                                                      self.neural_network,
                                                                                      self.neural_network_weights,
                                                                                      self.embedding_size,
                                                                                      self.clustering_loss_weight,
                                                                                      ssl_loss_weight,
                                                                                      self.custom_dataloaders,
                                                                                      self.augmentation_invariance,
                                                                                      self.initial_clustering_class,
                                                                                      initial_clustering_params,
                                                                                      self.device, random_state)
        self.labels_ = kmeans_labels
        self.cluster_centers_ = kmeans_centers
        self.dec_labels_ = dec_labels
        self.dec_cluster_centers_ = dec_centers
        self.neural_network_trained_ = neural_network
        self.n_features_in_ = X.shape[1]
        self.protocentroids_1 = protocentroids_1
        self.protocentroids_2 = protocentroids_2
        return self
