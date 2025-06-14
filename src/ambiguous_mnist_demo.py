# src/ambiguous_mnist_demo.py

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

    return X, X_full, y  # Return BOTH PCA and Original!


def match_average_confidence(model_class, X, target_conf, param_grid, is_vfkm):
    best_param = None
    best_gap = float('inf')

    for param in param_grid:
        if is_vfkm:
            model = model_class(n_clusters=10, lambda_entropy=param)
        else:
            model = model_class(n_clusters=10, temperature=param)

        model.fit(X)
        u = model.u
        avg_max = np.mean(np.max(u, axis=1))
        gap = abs(avg_max - target_conf)
        if gap < best_gap:
            best_gap = gap
            best_param = param

    return best_param


def try_fit_vfkm_for_ambiguity(X_pca, min_valid=5, max_attempts=10):
    """
    Incrementally search for a vfkm model with sufficient ambiguous examples.
    Returns best-fit model and its membership matrix.
    """
    for entropy_val in np.linspace(1.0, 10.0, max_attempts):
        model = RobustEntropyVFKM(n_clusters=10, lambda_entropy=entropy_val)
        model.fit(X_pca)
        u_vfkm = model.predict_proba(X_pca)

        min_clusters = 3
        min_thresh = 0.05
        max_prob = 0.95
        ambiguous = [
            i for i, row in enumerate(u_vfkm)
            if np.sum(row > min_thresh) >= min_clusters and np.max(row) < max_prob
        ]

        print(f"[λ={entropy_val:.2f}] Found {len(ambiguous)} ambiguous examples.")

        if len(ambiguous) >= min_valid:
            return model, u_vfkm

    raise RuntimeError(f"No entropy value found producing ≥{min_valid} ambiguous samples.")


def plot_soft_examples(X_full, y_true, u_vfkm, u_soft, threshold=0.95, num_samples=5):
    min_nonzero_clusters = 3
    min_membership_threshold = 0.05

    valid_indices = np.where([
        (np.sum(row > min_membership_threshold) >= min_nonzero_clusters) and (np.max(row) < threshold)
        for row in u_vfkm
    ])[0]

    if len(valid_indices) < num_samples:
        raise ValueError(
            f"Still too few valid ambiguous examples (found {len(valid_indices)}). "
            f"Try reducing `min_nonzero_clusters` or increasing `threshold`."
        )

    selected_indices = np.random.choice(valid_indices, size=num_samples, replace=False)

    fig, axs = plt.subplots(2, num_samples, figsize=(4 * num_samples, 6), sharey='row')

    for i, idx in enumerate(selected_indices):
        image = X_full[idx].reshape(28, 28)

        axs[0, i].imshow(image, cmap="gray")
        axs[0, i].axis('off')
        axs[0, i].set_title(
            f"Label: {y_true[idx]}\nVFKM={np.max(u_vfkm[idx]):.3f} | SoftKMeans={np.max(u_soft[idx]):.3f}",
            fontsize=10
        )

        x = np.arange(len(u_vfkm[idx]))

        axs[1, i].bar(x, u_vfkm[idx], color='royalblue', alpha=0.5, label='VFKM', width=0.4)
        axs[1, i].bar(x, u_soft[idx], color='coral', alpha=0.5, label='SoftKMeans', width=0.25)

        # Annotate vfkm bars (blue)
        for xi, val in enumerate(u_vfkm[idx]):
            if val > 1e-3:
                axs[1, i].text(
                    xi - 0.2, val * 0.8, f"{val:.3f}",
                    fontsize=8, ha='center', va='top', rotation=90, color='royalblue'
                )

        # Annotate SoftKMeans bars (red)
        for xi, val in enumerate(u_soft[idx]):
            if val > 1e-3:
                axs[1, i].text(
                    xi + 0.2, val * 0.8, f"{val:.3f}",
                    fontsize=8, ha='center', va='top', rotation=90, color='orangered'
                )



        axs[1, i].set_yscale('log')
        axs[1, i].set_ylim(1e-4, 1)

        if i == 0:
            axs[1, i].legend()
        else:
            axs[1, i].legend().remove()

    fig.text(0.5, 0.04, "Cluster Membership (log scale)", ha='center', fontsize=14)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def main():
    X_pca, X_full, y_true = load_mnist_subset(n_samples=5000, n_components=50)

    print("🔍 Matching average confidence...")

    soft_model = SoftKMeans(n_clusters=10, temperature=1.0)
    soft_model.fit(X_pca)
    avg_conf_soft = np.mean(np.max(soft_model.u, axis=1))
    print(f"[SoftKMeans] Avg Max Conf: {avg_conf_soft:.4f}")

    print("Searching for best VFKM entropy to expose ambiguity...")
    vfkm_model, u_vfkm = try_fit_vfkm_for_ambiguity(X_pca, min_valid=5)

    u_soft = soft_model.u

    print("Plotting soft examples...")
    plot_soft_examples(X_full, y_true, u_vfkm, u_soft, threshold=0.95, num_samples=5)


if __name__ == "__main__":
    main()
