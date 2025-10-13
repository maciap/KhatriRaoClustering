import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from KhatriRaoDeepClustering.deep_clustering_utils import HadamardAutoencoder
import numpy as np 
import torch.nn.functional as F
import torch
from clustpy.deep import detect_device 
from clustpy.deep.neural_networks import FeedforwardAutoencoder
from run_experiments_utils import closest_factors, shift_to_zero, load_data
from KRDeepExperimentsLib import run_deep_clustering_experiment



if __name__ == "__main__": 

    dataset = sys.argv[1] 
    random_state = 42
    batch_size = 512
    nrep = 20 
    embedding_size = 10 
    
    X, L = load_data(dataset) 
    L = shift_to_zero(L) # check that the minimum is zero otherwise translate 
    n_labels = len(np.unique(L))
    n_clusters1, n_clusters2 = closest_factors(n_labels)
      
    # for stickfigures, mnist, and double_mnist datasets, we divide by the maximum pixel for better performance, otherwise we standardize columns (z-scores) 
    if dataset!="stickfigures" and dataset!="mnist" and dataset!="double_mnist": 
        mean = np.mean(X, axis=0)        
        std = np.std(X, axis=0)          
        X = (X - mean) / (std + 1e-8)  # add epsilon to avoid division by zero
    else: 
        X = X / np.max(X)


    ''' train full autoencoder ''' 
    device = detect_device()    
    ae = FeedforwardAutoencoder(layers=[X.shape[1], 1024, 512, 256, embedding_size]).to(device)
    ae.fit(data=X, n_epochs=150, batch_size=batch_size, optimizer_params={"lr":1e-3})
    X_embed = ae.transform(X) 
    

    ''' train compressed autoencoder '''
    multiplier = 1 
    n_epochs = 1000
    had_ae = HadamardAutoencoder(layers=[X.shape[1], 1024, 512, 256, embedding_size], multiplier=multiplier).to(device)
    had_ae.fit(data=X, n_epochs=n_epochs, batch_size=batch_size, optimizer_params={"lr":1e-3})
    X_embed_had = had_ae.transform(X)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    X_reconstructed_tensor =  ae(X_tensor) 
    loss = F.mse_loss(X_reconstructed_tensor, X_tensor)
    X_reconstructed_tensor_had =  had_ae(X_tensor)
    loss_had = F.mse_loss(X_reconstructed_tensor_had, X_tensor)
    while loss_had > loss and multiplier<10: 
        multiplier+=1 
        n_epochs+=500 
        had_ae = HadamardAutoencoder(layers=[X.shape[1], 1024, 512, 256, embedding_size], multiplier=multiplier).to(device)
        had_ae.fit(data=X, n_epochs=n_epochs, batch_size=batch_size, optimizer_params={"lr":1e-3})
        X_embed_had = had_ae.transform(X)
        X_reconstructed_tensor_had =  had_ae(X_tensor) 
        loss_had = F.mse_loss(X_reconstructed_tensor_had, X_tensor)

    total_params_ae = sum(p.numel() for p in ae.parameters())
    total_params_had_ae =  sum(p.numel() for p in had_ae.parameters())

    full_centroid_params = embedding_size * (n_clusters1 * n_clusters2)
    kr_centroid_params = embedding_size * (n_clusters1 + n_clusters2)
    
    total_params_kr_clustering = total_params_had_ae + kr_centroid_params
    total_params_standard_clustering = total_params_ae + full_centroid_params

    # KR IDEC
    best_metrics_kr_dec = run_deep_clustering_experiment("KR IDEC",  n_clusters1,  n_clusters2, had_ae, ae,  nrep, batch_size)

    # IDEC  
    best_metrics_idec = run_deep_clustering_experiment("IDEC",  n_clusters1,  n_clusters2, had_ae, ae,  nrep, batch_size)

    # KR DKM 
    best_metrics_kr_dkm = run_deep_clustering_experiment("KR DKM",  n_clusters1,  n_clusters2, had_ae, ae,  nrep, batch_size)

    # DKM 
    best_metrics_dkm = run_deep_clustering_experiment("DKM",  n_clusters1,  n_clusters2, had_ae, ae,  nrep, batch_size)

