## Variational Fuzzy K-Means (VFKM)

A principled soft clustering framework derived from variational free energy minimization. This repository implements VFKM along with its ablation variants and baseline clustering algorithms. The goal is to enable robust and interpretable fuzzy clustering via entropy regularization, KL anchoring, and variational inference.

### Installation

#### Clone the repo :
- git clone https://github.com/abhisheksharma026/variational-fuzzy-kmeans.git
- cd variational-fuzzy-kmeans

#### Set up the virtual environment :
- poetry install

#### Running the Benchmark : Run all models (baselines + VFKM + ablations) on multiple datasets with 5-fold cross-validation
- poetry run python src/runner.py

#### Testing : Run unit tests
- poetry run pytest tests/

#### Key Observations
Breast Cancer:
GMM and Agglomerative slightly outperform VFKM in ARI and NMI, while Soft-KMeans and VFKM achieve the best silhouette scores—indicating compactness.

Digits:
VFKM (No Anneal) achieves the highest ARI (0.5029), but NMI remains best for Agglomerative. VFKM performs competitively on Silhouette and Gower.

USPS:
Agglomerative dominates in ARI and NMI, while VFKM and Soft-KMeans tie for top Silhouette and Gower values.

MNIST:
Agglomerative again leads in ARI and NMI. VFKM variants and Soft-KMeans outperform GMM in compactness metrics, proving beneficial in high-dimensional fuzzy clustering.

VFKM Ablations:
Disabling entropy or KL often results in faster convergence but sacrifices stability and interpretability. The full VFKM model achieves balanced performance across metrics.

