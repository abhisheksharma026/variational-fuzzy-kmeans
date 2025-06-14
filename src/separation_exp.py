# src/toy_overlap_demo.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from model import AnnealedSoftKMeans, RobustEntropyVFKM


def generate_overlap_blobs(n_samples=1000, centers=5, cluster_std=5.0, random_state=42):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )
    X = StandardScaler().fit_transform(X)
    return X, y


def plot_soft_assignments_heatmaps(X, u_soft, u_vfkm, title_soft, title_vfkm):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)


    # Sort by top-1 cluster for visual grouping
    order_soft = np.argsort(np.argmax(u_soft, axis=1))
    order_vfkm = np.argsort(np.argmax(u_vfkm, axis=1))

    vmin = 0.0
    vmax = 1.0

    im0 = axs[0].imshow(u_soft[order_soft], aspect='auto', cmap='plasma', vmin=vmin, vmax=vmax)
    axs[0].set_title(title_soft)
    axs[0].set_xlabel("Clusters")
    axs[0].set_ylabel("Samples")

    im1 = axs[1].imshow(u_vfkm[order_vfkm], aspect='auto', cmap='plasma', vmin=vmin, vmax=vmax)
    axs[1].set_title(title_vfkm)
    axs[1].set_xlabel("Clusters")
    axs[1].set_ylabel("Samples")

    cbar = fig.colorbar(im1, ax=axs, orientation="vertical", fraction=0.015, pad=0.03)
    cbar.set_label("Membership Probability", rotation=270, labelpad=15)

    axs[0].set_xticks(np.arange(u_soft.shape[1]))
    axs[1].set_xticks(np.arange(u_vfkm.shape[1]))

    plt.show()


def main():
    X, y = generate_overlap_blobs(n_samples=1000)

    print("🔵 Training Annealed Soft KMeans...")
    soft_model = AnnealedSoftKMeans(n_clusters=5, init_temp=5.0, final_temp=0.5, max_iter=1000)
    soft_model.fit(X)
    u_soft = soft_model.u

    print("🔵 Training Entropy-VFKM (KL+Anneal)...")
    vfkm_model = RobustEntropyVFKM(
        n_clusters=5,
        lambda_kl=0.5,
        anneal_gamma=0.05,
        max_iter=1000,
        verbose=False
    )
    vfkm_model.fit(X)
    u_vfkm = vfkm_model.predict_proba(X)

    print("✅ Plotting soft assignment heatmaps...")
    plot_soft_assignments_heatmaps(
        X, u_soft, u_vfkm,
        title_soft="Annealed Soft KMeans - Soft Membership",
        title_vfkm="Entropy-vfkm (KL+Anneal) - Soft Membership"
    )


if __name__ == "__main__":
    main()
