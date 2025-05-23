import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from model import RobustEntropyVFKM

def load_mnist_pca(n_components=100, random_state=42):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X_full, y = mnist['data'], mnist['target'].astype(int)
    X_scaled = StandardScaler().fit_transform(X_full)
    X_pca = PCA(n_components=n_components, random_state=random_state).fit_transform(X_scaled)
    return X_pca, y

def train_vfkm_and_track(X, params, n_clusters=10, max_iter=100):
    model = RobustEntropyVFKM(n_clusters=n_clusters, max_iter=max_iter, **params, verbose=False)
    model.fit(X)
    return model.loss_history if hasattr(model, 'loss_history') else None

X_mnist, y_mnist = load_mnist_pca(n_components=100)

variants = {
    "Full VFKM": {"lambda_entropy": 5.0, "lambda_kl": 0.5, "anneal_gamma": 0.02},
    "No Entropy": {"lambda_entropy": 1e-5, "lambda_kl": 0.5, "anneal_gamma": 0.02},
    "No KL": {"lambda_entropy": 5.0, "lambda_kl": 0.0, "anneal_gamma": 0.02},
    "No Anneal": {"lambda_entropy": 5.0, "lambda_kl": 0.5, "anneal_gamma": 0.0},
    "No Entropy + No KL": {"lambda_entropy": 1e-5, "lambda_kl": 0.0, "anneal_gamma": 0.0}
}

convergence_results = {}
for name, params in variants.items():
    print(f"Training {name}...")
    convergence_results[name] = train_vfkm_and_track(X_mnist, params, max_iter=200)

plt.figure(figsize=(12, 6))
for name, losses in convergence_results.items():
    if losses is not None:
        plt.plot(losses, label=name)

plt.xlabel("Iteration")
plt.ylabel("Free Energy Loss")
plt.title("Convergence of VFKM Ablation Variants on MNIST")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
