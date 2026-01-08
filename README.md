# Khatri-Rao Clustering

This code implements the Khatri-Rao clustering algorithms introduced in the paper "Khatri-Rao Clustering for Data Summarization". 
The code is written in Python 3. 

Khatri-Rao clustering is a paradigm extending existing centroid-based clustering algorithms to achieve higher compression rates.  Khatri-Rao clustering algorithms can find more succinct yet equally accurate data summaries compared to standard clustering algorithms. 

This code implements the extensions to the Khatri-Rao paradigm of the standard k-means algorithm and of two deep clustering algorithms, deep k-means (DKM), and improved deep embedded clustering (IDEC). 

![Example](stickfigures.png)


## 🔧 Install
- Create Conda environment:  
  ```bash
  conda env create -f kr.yml
  ```


## 📁 Repository contents 
- `KathriRaokMeans/`
   - `kr_k_means_space_efficient.py`: space-efficient implementation of the Khatri-Rao k-means algorithm (default). 
   - `kr_k_means_time_efficient.py`: time-efficient implementation of the Khatri-Rao k-means algorithm. 
   - `kr_k_means_p_sets.py`: implements the Khatri-Rao k-means algorithm with an arbitrary number of protocentroids. 
   - `kr_k_means_utils.py`: utilities for the Khatri-Rao k-means algorithm.

- `KhatriRaoDeepClustering/`
  - `kr_dkm.py`: implements the Khatri-Rao DKM deep clustering algorithm. 
  - `kr_idec.py`: implements the Khatri-Rao IDEC deep clustering algorithm.
  - `deep_clustering_utils.py`: utilities for Khatri-Rao deep clustering algorithms.

- `Scripts/`
  - `run_k_means_experiments.py`: run experiments comparing standard and Khatri-Rao k-means clustering.
  - `run_deep_clustering_experiments.py`: run experiments comparing standard and Khatri-Rao deep clustering.
  - `run_experiments_utils.py`: general utilities for all experiments.
  - `KRkmeansExperimentsLib.py`: utilities for k-means experiments.
  - `KRDeepExperimentsLib.py`: utilities for deep clustering experiments.
  - `run_k_means_experiments_by_p`: run experiments comparing standard and Khatri-Rao k-means clustering for different numbers of sets of protocentroids. 
  - `KRkmeansExperimentsLib_p_sets.py`: utilities for k-means experiments with arbitrary numbers of protocentroids.
  - `run_scalability_experiments`: run scalability analysis for Khatri-Rao k-means clustering and standard k-means. 
  - `KRkmeansExperimentsLibScalability.py`: utilities for scalability analysis.

- `notebooks/`
  - `HardClusteringViaMatrixDecomposition`: example notebook showcasing an approach to Khatri-Rao clustering based on matrix decomposition. 



- 📎`Appendix.pdf`: appendix containing additional dataset descriptions, implementation details and additional technical details.

## ✏️ Minimal example
```python
from kr_k_means_space_efficient import KrKMeans
X = np.random.randn(250, 2) # Full-rank 250 x 250 matrix with i.i.d standard gaussian entries
n_protocentroids_set_1, n_protocentroids_set_2 = 3, 3
kr_kmeans = KrKMeans(X, n_protocentroids_set_1, n_protocentroids_set_2, operator=operator)
_, kr_k_means_loss, _ = kr_kmeans.fit(n_iter=100, init_type="random", th_movement=0.0001)
print(f"The inertia of the clustering solution with two sets of {n_protocentroids_set_1} and {n_protocentroids_set_2} protocentroids is {kr_k_means_loss}")
```

## ▶️ Running scripts from the command line
```bash 
# Run Khatri–Rao k-Means (space-efficient implementation)
python run_kr_k_means.py --dataset stickfigures --impl space --n_iter 200
```

```bash 
# Run scripts to reproduce k-means experiments (and similarly for the other experiments)
python .\scripts\run_k_means_experiments.py stickfigures
```


## 📚 Datasets

The datasets used in the experiments are available through the following open-source Python libraries:
- **[scikit-learn](https://scikit-learn.org/stable/datasets.html)**
- **[ClustPy](https://clustpy.readthedocs.io/)** 
- **[ClustBench](https://clustering-benchmarks.gagolewski.com/)** 

All ClustPy datasets are fetched automatically at runtime.
On the other hand, for the <small><b>r15</b></small> and <small><b>chameleon</b></small> datasets that are available in ClustBench, one needs to [download](https://github.com/gagolews/clustering-data-v1/releases/tag/v1.1.0/) the Benchmark Suite (v1.1.0) repository onto their own disk and place it within the `Scripts/` folder.  
