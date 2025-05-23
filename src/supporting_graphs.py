import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from model import SoftKMeans, RobustEntropyVFKM


def load_mnist_subset(n_samples=5000, n_components=50, random_state=42):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X_full, y = mnist['data'], mnist['target'].astype(int)
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X_full), size=n_samples, replace=False)

    X_full = X_full[indices]
    y = y[indices]

    # Preprocessing for clustering (PCA)
    X = StandardScaler().fit_transform(X_full)
    X = PCA(n_components=n_components, random_state=random_state).fit_transform(X)

    return X, y


X_pca, y_true = load_mnist_subset()

soft_model = SoftKMeans(n_clusters=10, temperature=1.0)
soft_model.fit(X_pca)

bfkm_model = RobustEntropyVFKM(n_clusters=10, lambda_entropy=1.5)
bfkm_model.fit(X_pca)

u_soft = soft_model.u
u_bfkm = bfkm_model.predict_proba(X_pca)

conf_bfkm = np.max(u_bfkm, axis=1)
conf_soft = np.max(u_soft, axis=1)

entropy_bfkm = -np.sum(u_bfkm * np.log(u_bfkm + 1e-10), axis=1)
entropy_soft = -np.sum(u_soft * np.log(u_soft + 1e-10), axis=1)

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

axs[0].hist(conf_bfkm, bins=30, alpha=0.6, label='BFKM', color='royalblue')
axs[0].hist(conf_soft, bins=30, alpha=0.6, label='SoftKMeans', color='orangered')
axs[0].set_title("Distribution of Max Membership Confidence")
axs[0].set_xlabel("Max $u_{ik}$")
axs[0].set_ylabel("Frequency")
axs[0].legend()

axs[1].hist(entropy_bfkm, bins=30, alpha=0.6, label='BFKM', color='royalblue')
axs[1].hist(entropy_soft, bins=30, alpha=0.6, label='SoftKMeans', color='orangered')
axs[1].set_title("Distribution of Membership Entropy")
axs[1].set_xlabel("Entropy $H(u_i)$")
axs[1].set_ylabel("Frequency")
axs[1].legend()

plt.tight_layout()
plt.show()
