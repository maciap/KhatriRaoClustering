import numpy as np
import torch
from clustpy.deep._utils import encode_batchwise, predict_batchwise, mean_squared_error, squared_euclidean_distance
from clustpy.deep._train_utils import get_default_deep_clustering_initialization
from clustpy.deep._abstract_deep_clustering_algo import _AbstractDeepClusteringAlgo
from sklearn.cluster import KMeans
from sklearn.base import ClusterMixin
from collections.abc import Callable
from clustpy.deep.dkm import _DKM_Module, _dkm_get_probs, _get_default_alphas
import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from KathriRaokMeans.kr_k_means import KrKMeans
from clustpy.utils import plot_with_transformation
import tqdm


def _kr_dkm(X: np.ndarray, n_clusters_1: int, n_clusters_2: int, alphas: list | tuple, batch_size: int, pretrain_optimizer_params: dict,
         clustering_optimizer_params: dict, pretrain_epochs: int, clustering_epochs: int,
         optimizer_class: torch.optim.Optimizer, ssl_loss_fn: Callable | torch.nn.modules.loss._Loss,
         neural_network: torch.nn.Module | tuple, neural_network_weights: str, embedding_size: int,
         clustering_loss_weight: float, ssl_loss_weight: float, custom_dataloaders: tuple,
         augmentation_invariance: bool, initial_clustering_class: ClusterMixin, initial_clustering_params: dict,
         device: torch.device, random_state: np.random.RandomState) -> (
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module):
    """
    Start the actual DKM clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray / torch.Tensor
        the given data set. Can be a np.ndarray or a torch.Tensor
    n_clusters_1 : int
        number of protocentroids in set 1 
    n_clusters_2 : int
        number of protocentroids in set 2 
    alphas : list | tuple
        Small values close to 0 are equivalent to homogeneous assignments to all clusters. Large values simulate a clear assignment as with kMeans.
        list of different alpha values used for the prediction
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
        _, dkm_loss, _ = kr_kmeans.fit(100, init_type="random", th_movement=0.0001)
        if dkm_loss < best_score:
            print(f"better result in iteration {init_iter} with loss {dkm_loss}")
            best_score = dkm_loss
            best_krkmeans = kr_kmeans
    init_labels = best_krkmeans.predict(embedded_data_init)
    dkm_module = _KR_DKM_Module(best_krkmeans.B_1, best_krkmeans.B_2, operator, alphas, augmentation_invariance).to(device)
    optimizer = optimizer_class(list(neural_network.parameters()), **clustering_optimizer_params)
    optimizer_centroids = optimizer_class(list(dkm_module.parameters()),
                                **{"lr":clustering_optimizer_params["lr"]*10})
    # DKM Training loop
    dkm_module.fit(neural_network, trainloader, clustering_epochs, device, optimizer, ssl_loss_fn,
                   clustering_loss_weight, ssl_loss_weight, optimizer_centroids)
    # Get labels
    dec_labels = predict_batchwise(testloader, neural_network, dkm_module)
    dec_centers = dkm_module._get_centers().detach().cpu().numpy()
    # Do reclustering with Kmeans
    embedded_data = encode_batchwise(testloader, neural_network)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embedded_data)
    return kmeans.labels_, kmeans.cluster_centers_, dec_labels, dec_centers, neural_network, dkm_module.protocentroids_1.detach().cpu().numpy(), dkm_module.protocentroids_2.detach().cpu().numpy()


class _KR_DKM_Module(_DKM_Module):
    """
    The _KR_DKM_Module. Contains most of the algorithm specific procedures like the loss and prediction functions.

    Parameters
    ----------
    protocentroids_1 : np.ndarray
        The initial protocentroids in set 1 
    protocentroids_2 : np.ndarray
        The initial protocentroids in set 2  
    alphas : list
        list of different alpha values used for the prediction
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

    def __init__(self, protocentroids_1, protocentroids_2, operator: str, alphas: float, augmentation_invariance: bool = False):
        super().__init__(np.array(0.), alphas, augmentation_invariance)
        assert operator in ["product", "sum"], f"operator must be 'sum' or 'product' but is {operator}"
        self.operator = operator
        # Centers are learnable parameters
        self.protocentroids_1 = torch.nn.Parameter(torch.tensor(protocentroids_1), requires_grad=True)
        self.protocentroids_2 = torch.nn.Parameter(torch.tensor(protocentroids_2), requires_grad=True)

    def _get_centers(self):
        # Reshape tensors B_1 and B_2 to allow broadcasting
        dim = self.protocentroids_1.shape[1]

        reshaped_B_1 = self.protocentroids_1.view(self.protocentroids_1.shape[0], 1, dim)
        reshaped_B_2 = self.protocentroids_2.view(1, self.protocentroids_2.shape[0], dim)
        # Element-wise multiplication and reshaping
        if self.operator == "product":
            allcenters = (reshaped_B_1 * reshaped_B_2).view(self.protocentroids_1.shape[0] * self.protocentroids_2.shape[0], dim)       
        else:
            allcenters = (reshaped_B_1 + reshaped_B_2).view(self.protocentroids_1.shape[0] * self.protocentroids_2.shape[0], dim)    
        return allcenters

    def predict(self, embedded: torch.Tensor, alpha: float = 1000) -> torch.Tensor:
        """
        Soft prediction of given embedded samples. Returns the corresponding soft labels.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        alpha : float
            the alpha value (default: 1000)

        Returns
        -------
        pred : torch.Tensor
            The predicted soft labels
        """
        centers = self._get_centers()
        squared_diffs = squared_euclidean_distance(embedded, centers)
        pred = _dkm_get_probs(squared_diffs, alpha)
        return pred

    def dkm_loss(self, embedded: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Calculate the DKM loss of given embedded samples and given alpha.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        alpha : float
            the alpha value

        Returns
        -------
        loss : torch.Tensor
            the final DKM loss
        """
        centers = self._get_centers()
        squared_diffs = squared_euclidean_distance(embedded, centers)
        probs = _dkm_get_probs(squared_diffs, alpha)
        loss = (squared_diffs * probs).sum(1).mean()
        return loss

    def dkm_augmentation_invariance_loss(self, embedded: torch.Tensor, embedded_aug: torch.Tensor,
                                         alpha: float) -> torch.Tensor:
        """
        Calculate the DKM loss of given embedded samples with augmentation invariance and given alpha.

        Parameters
        ----------
        embedded : torch.Tensor
            the embedded samples
        embedded_aug : torch.Tensor
            the embedded augmented samples
        alpha : float
            the alpha value

        Returns
        -------
        loss : torch.Tensor
            the final DKM loss
        """
        # Get loss of non-augmented data
        centers = self._get_centers()
        squared_diffs = squared_euclidean_distance(embedded, centers)
        probs = _dkm_get_probs(squared_diffs, alpha)
        clean_loss = (squared_diffs * probs).sum(1).mean()
        # Get loss of augmented data
        squared_diffs_augmented = squared_euclidean_distance(embedded_aug, centers)
        aug_loss = (squared_diffs_augmented * probs).sum(1).mean()
        # average losses
        loss = (clean_loss + aug_loss) / 2
        return loss
    
    def fit(self, neural_network: torch.nn.Module, trainloader: torch.utils.data.DataLoader, n_epochs: int,
            device: torch.device, optimizer: torch.optim.Optimizer, ssl_loss_fn: Callable | torch.nn.modules.loss._Loss,
            clustering_loss_weight: float, ssl_loss_weight: float, optimizer_centroids) -> '_DKM_Module':
        """
        Trains the _KR_DKM_Module in place.

        Parameters
        ----------
        neural_network : torch.nn.Module
            the neural network
        trainloader : torch.utils.data.DataLoader
            dataloader to be used for training
        n_epochs : int
            number of epochs for the clustering procedure.
            The total number of epochs therefore corresponds to: len(alphas)*n_epochs
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
        self : _KR_DKM_Module
            this instance of the _KR_DKM_Module
        """
        tbar = tqdm.tqdm(total=n_epochs * len(self.alphas), desc="DKM training")
        for alpha in self.alphas:
            for _ in range(n_epochs):
                total_loss = 0
                for batch in trainloader:
                    loss = self._loss(batch, alpha, neural_network, clustering_loss_weight, ssl_loss_weight,
                                      ssl_loss_fn, device)
                    total_loss += loss.item()
                    # Backward pass
                    optimizer.zero_grad()
                    optimizer_centroids.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optimizer_centroids.step()
                postfix_str = {"Loss": total_loss, "Alpha": alpha}
                tbar.set_postfix(postfix_str)
                tbar.update()
        return self

class KR_DKM(_AbstractDeepClusteringAlgo):
    """
    A Kathri Rao variant of DKM.
    First, a neural network will be trained (will be skipped if input neural network is given).
    Afterward, KMeans identifies the initial clusters.
    Last, the network will be optimized using the DKM loss function.

    Parameters
    ----------
    n_clusters_1 : int
        number of protocentroids in set 1
    n_clusters_2: int 
        number of protocentroids in set 2
    alphas : tuple
        tuple of different alpha values used for the prediction.
        Small values close to 0 are equivalent to homogeneous assignments to all clusters. Large values simulate a clear assignment as with kMeans.
        If None, the default calculation of the paper will be used.
        This is equal to \alpha_{i+1}=2^{1/log(i)^2}*\alpha_i with \alpha_1=0.1 and maximum i=40.
        Alpha can also be a tuple with (None, \alpha_1, maximum i) (default: (1000))
    batch_size : int
        size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the neural network, includes the learning rate. If None, it will be set to {"lr": 1e-3} (default: None)
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate. If None, it will be set to {"lr": 1e-4} (default: None)
    pretrain_epochs : int
        number of epochs for the pretraining of the neural network (default: 100)
    clustering_epochs : int
        number of epochs for each alpha value for the actual clustering procedure.
        The total number of clustering epochs therefore corresponds to: len(alphas)*clustering_epochs (default: 150)
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
        weight of the clustering loss (default: 0.1)
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
    dkm_labels_ : np.ndarray
        The final DKM labels
    dkm_cluster_centers_ : np.ndarray
        The final DKM cluster centers
    neural_network_trained_ : torch.nn.Module
        The final neural network
    n_features_in_ : int
        the number of features used for the fitting

    

    References
    ----------
    Fard, Maziar Moradi, Thibaut Thonet, and Eric Gaussier. "Deep k-means: Jointly clustering with k-means and learning representations."
    Pattern Recognition Letters 138 (2020): 185-192.
    """

    def __init__(self, n_clusters_1: int = 8, n_clusters_2: int = None, alphas: tuple = (1000), batch_size: int = 256,
                 pretrain_optimizer_params: dict = None, clustering_optimizer_params: dict = None,
                 pretrain_epochs: int = 100, clustering_epochs: int = 150,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 ssl_loss_fn: Callable | torch.nn.modules.loss._Loss = mean_squared_error,
                 neural_network: torch.nn.Module | tuple = None, neural_network_weights: str = None,
                 embedding_size: int = 10, clustering_loss_weight: float = 0.1, ssl_loss_weight: float = 1.,
                 custom_dataloaders: tuple = None, augmentation_invariance: bool = False,
                 initial_clustering_class: ClusterMixin = KMeans, initial_clustering_params: dict = None,
                 device: torch.device = None, random_state: np.random.RandomState | int = None):
        super().__init__(batch_size, neural_network, neural_network_weights, embedding_size, device, random_state)
        self.n_clusters_1 = n_clusters_1
        self.n_clusters_2 = n_clusters_2
        self.alphas = alphas
        self.pretrain_optimizer_params = pretrain_optimizer_params
        self.clustering_optimizer_params = clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.optimizer_class = optimizer_class
        self.ssl_loss_fn = ssl_loss_fn
        self.clustering_loss_weight = clustering_loss_weight
        self.ssl_loss_weight = ssl_loss_weight
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.initial_clustering_class = initial_clustering_class
        self.initial_clustering_params = initial_clustering_params

    def _check_alphas(self) -> list:
        """
        Compute the actual alphas.

        Returns
        -------
        alphas : list
            the list with the alpha values
        """
        alphas = self.alphas
        if alphas is None:
            alphas = _get_default_alphas()
        elif (type(alphas) is tuple or type(alphas) is list) and len(alphas) == 3 and alphas[0] is None:
            alphas = _get_default_alphas(init_alpha=alphas[1], n_alphas=alphas[2])
        elif type(alphas) is int or type(alphas) is float:
            alphas = [alphas]
        assert type(alphas) is tuple or type(alphas) is list, "alphas must be a list, int or tuple"
        return alphas

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'KR_DKM':
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
        self : KR_DKM
            this instance of the KR_DKM algorithm
        """
        X, _, random_state, pretrain_optimizer_params, clustering_optimizer_params, initial_clustering_params = self._check_parameters(X, y=y)
        alphas = self._check_alphas()
        if self.n_clusters_2 is None:
            assert np.sqrt(self.n_clusters_1) % 1 == 0
            n_clusters_1 = int(np.sqrt(self.n_clusters_1))
            n_clusters_2 = n_clusters_1
        else:
            n_clusters_1 = self.n_clusters_1
            n_clusters_2 = self.n_clusters_2
        kmeans_labels, kmeans_centers, dkm_labels, dkm_centers, neural_network, protocentroids_1, protocentroids_2 = _kr_dkm(X, n_clusters_1, n_clusters_2, alphas,
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
                                                                                      self.ssl_loss_weight,
                                                                                      self.custom_dataloaders,
                                                                                      self.augmentation_invariance,
                                                                                      self.initial_clustering_class,
                                                                                      initial_clustering_params,
                                                                                      self.device,
                                                                                      random_state)
        self.labels_ = kmeans_labels
        self.cluster_centers_ = kmeans_centers
        self.dkm_labels_ = dkm_labels
        self.dkm_cluster_centers_ = dkm_centers
        self.neural_network_trained_ = neural_network
        self.set_n_featrues_in(X.shape[1])
        self.protocentroids_1 = protocentroids_1
        self.protocentroids_2 = protocentroids_2
        return self
