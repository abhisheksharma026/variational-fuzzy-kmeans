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
- GMM and Agglomerative slightly outperform VFKM in ARI and NMI
- While Soft-KMeans and VFKM achieve the best silhouette scores—indicating compactness.

Digits:
- VFKM (No Anneal) achieves the highest ARI (0.5029)
- NMI remains best for Agglomerative.
- VFKM performs competitively on Silhouette and Gower.

USPS:
- Agglomerative dominates in ARI and NMI
- While VFKM and Soft-KMeans tie for top Silhouette and Gower values.

MNIST:
- Agglomerative again leads in ARI and NMI.
- VFKM variants and Soft-KMeans outperform GMM in compactness metrics, proving beneficial in high-dimensional fuzzy clustering.

VFKM Ablations:
- Disabling entropy or KL often results in faster convergence but sacrifices stability and interpretability.
- The full VFKM model achieves balanced performance across metrics.

---

**Overall Results:**

<details> <summary><strong> Breast Cancer</strong></summary>

| Model                     | ARI        | NMI        | Silhouette | Weighted Gower |
| ------------------------- | ---------- | ---------- | ---------- | -------------- |
| KMeans                    | 0.6531     | 0.5596     | 0.3497     | 0.1546         |
| **GMM**                   | **0.6812** | 0.5876     | 0.3491     | **0.1545**     |
| Agglomerative             | 0.6665     | **0.6008** | 0.3378     | 0.1571         |
| Soft-KMeans               | 0.6419     | 0.5500     | **0.3517** | 0.1547         |
| Annealed-Soft-KMeans      | 0.6419     | 0.5500     | **0.3517** | 0.1547         |
| VFKM (No Entropy)         | 0.6366     | 0.5454     | 0.3514     | 0.1547         |
| VFKM (No KL)              | 0.6414     | 0.5470     | 0.3503     | 0.1546         |
| VFKM (No Anneal)          | 0.6419     | 0.5500     | **0.3517** | 0.1547         |
| VFKM (No Entropy + No KL) | 0.6366     | 0.5454     | 0.3514     | 0.1547         |
| VFKM                      | 0.6419     | 0.5500     | **0.3517** | 0.1547         |
</details>

<details> <summary><strong> Digits</strong></summary>

| Model                     | ARI        | NMI        | Silhouette | Weighted Gower |
| ------------------------- | ---------- | ---------- | ---------- | -------------- |
| KMeans                    | 0.4495     | 0.6252     | 0.1403     | 0.1723         |
| GMM                       | 0.4804     | 0.6481     | 0.1377     | **0.1704**     |
| Agglomerative             | 0.4982     | **0.6998** | 0.1247     | 0.1770         |
| Soft-KMeans               | 0.4914     | 0.6688     | 0.1421     | 0.1711         |
| Annealed-Soft-KMeans      | 0.5013     | 0.6764     | **0.1435** | 0.1707         |
| VFKM (No Entropy)         | 0.4873     | 0.6537     | 0.1407     | 0.1720         |
| VFKM (No KL)              | 0.4969     | 0.6662     | 0.1427     | 0.1711         |
| **VFKM (No Anneal)**      | **0.5029** | 0.6773     | 0.1433     | 0.1708         |
| VFKM (No Entropy + No KL) | 0.4873     | 0.6537     | 0.1407     | 0.1720         |
| VFKM                      | 0.5021     | 0.6772     | 0.1434     | 0.1709         |
</details>

<details> <summary><strong> USPS</strong></summary>

| Model                     | ARI        | NMI        | Silhouette | Weighted Gower |
| ------------------------- | ---------- | ---------- | ---------- | -------------- |
| KMeans                    | 0.4698     | 0.5782     | 0.1452     | **0.1149**     |
| GMM                       | 0.4417     | 0.5531     | 0.1449     | 0.1150         |
| **Agglomerative**         | **0.5350** | **0.6551** | 0.1209     | 0.1149         |
| Soft-KMeans               | 0.4605     | 0.5693     | 0.1462     | 0.1149         |
| Annealed-Soft-KMeans      | 0.4577     | 0.5679     | **0.1464** | 0.1149         |
| VFKM (No Entropy)         | 0.4616     | 0.5703     | 0.1460     | 0.1149         |
| VFKM (No KL)              | 0.4578     | 0.5682     | 0.1462     | 0.1149         |
| VFKM (No Anneal)          | 0.4600     | 0.5702     | 0.1461     | 0.1149         |
| VFKM (No Entropy + No KL) | 0.4616     | 0.5703     | 0.1460     | 0.1149         |
| VFKM                      | 0.4596     | 0.5694     | 0.1462     | 0.1149         |
</details>

<details> <summary><strong> MNIST</strong></summary>

| Model                     | ARI        | NMI        | Silhouette | Weighted Gower |
| ------------------------- | ---------- | ---------- | ---------- | -------------- |
| KMeans                    | 0.3021     | 0.4168     | **0.0446** | **0.0350**     |
| GMM                       | 0.2583     | 0.3835     | -0.0228    | 0.0357         |
| **Agglomerative**         | **0.4026** | **0.5744** | -0.0089    | 0.0360         |
| Soft-KMeans               | 0.2973     | 0.4127     | 0.0418     | 0.0351         |
| Annealed-Soft-KMeans      | 0.2966     | 0.4125     | 0.0423     | 0.0351         |
| VFKM (No Entropy)         | 0.2970     | 0.4120     | 0.0418     | 0.0351         |
| VFKM (No KL)              | 0.2964     | 0.4112     | 0.0421     | 0.0351         |
| VFKM (No Anneal)          | 0.2969     | 0.4127     | 0.0421     | 0.0351         |
| VFKM (No Entropy + No KL) | 0.2970     | 0.4120     | 0.0418     | 0.0351         |
| VFKM                      | 0.2976     | 0.4134     | 0.0418     | 0.0351         |
</details>

---

<img width="1510" alt="Research" src="https://github.com/user-attachments/assets/7aaa3bbc-e2ab-456d-a6dd-36b81b02be76" />


---
If you use this work, please cite it as: 

```
@misc{sharma_2025_variational_fuzzy_kmeans,
  author       = {Abhishek Sharma},
  title        = {Variational Fuzzy K-Means: A Free Energy-Based Approach},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15667345},
  url          = {https://doi.org/10.5281/zenodo.15667345}
}
```
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15667345.svg)](https://doi.org/10.5281/zenodo.15667345)
