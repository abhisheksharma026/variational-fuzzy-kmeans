import numpy as np
import pandas as pd
import time
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

def load_digits_data():
    X, y = load_digits(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return X, y

def load_breast_cancer_data():
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return X, y

def load_mnist_data(sample_size=70000, n_components=100, random_state=42):
    """
    Loads a small sample of MNIST with PCA compression.

    Args:
        sample_size (int): Number of samples.
        n_components (int): PCA components.
        random_state (int): RNG seed.

    Returns:
        X (np.ndarray): Features
        y (np.ndarray): Labels
    """
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data']
    y = mnist['target'].astype(int)

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X), size=sample_size, replace=False)
    X = X[indices]
    y = y[indices]
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=random_state)
    X = pca.fit_transform(X)

    return X, y


def load_usps_data(sample_size=9298, n_components=256, random_state=42):
    """
    Load a subsample of USPS dataset, apply StandardScaler + PCA.

    Args:
        sample_size (int): Number of samples to select
        n_components (int): Number of PCA components
        random_state (int): Random seed

    Returns:
        X (np.ndarray): Preprocessed features
        y (np.ndarray): Integer labels
    """
    usps = fetch_openml('usps', version=2, as_frame=False, parser='auto')
    X = usps['data']
    y = usps['target'].astype(int)

    # Subsample
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X), size=sample_size, replace=False)
    X = X[indices]
    y = y[indices]
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=random_state)
    X = pca.fit_transform(X)

    return X, y


def compute_confidence(u_matrix):
    """
    Computes mean confidence from the membership matrix.
    
    Args:
        u_matrix (np.ndarray): Membership matrix (n_samples, n_clusters).

    Returns:
        float: Mean maximum membership per sample.
    """
    return np.mean(np.max(u_matrix, axis=1))

def compute_kmeans_confidence(kmeans_model, X):
    """
    Approximate confidence for KMeans based on softmax over negative distances.
    """
    distances = np.linalg.norm(X[:, None, :] - kmeans_model.cluster_centers_[None, :, :], axis=2)  # (n_samples, n_clusters)
    logits = -distances
    logits -= logits.max(axis=1, keepdims=True)  # Numerical stability
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    confidence = np.mean(np.max(probs, axis=1))
    return confidence

# --- 🆕 Gower Distance Helper
def compute_weighted_gower_score(X, labels):
    """
    Compute weighted average intra-cluster Gower distance.

    Args:
        X (np.ndarray): Data matrix.
        labels (np.ndarray): Cluster labels.

    Returns:
        float: Weighted Gower distance.
    """
    from sklearn.preprocessing import MinMaxScaler
    X_scaled = MinMaxScaler().fit_transform(X)

    gower_per_cluster = []
    cluster_sizes = []

    for cluster_id in np.unique(labels):
        cluster_points = X_scaled[labels == cluster_id]
        if len(cluster_points) <= 1:
            continue
        dists = cdist(cluster_points, cluster_points, metric="cityblock") / cluster_points.shape[1]
        avg_dist = np.mean(dists)
        gower_per_cluster.append(avg_dist)
        cluster_sizes.append(len(cluster_points))

    weighted_gower = np.average(gower_per_cluster, weights=cluster_sizes)
    return weighted_gower

# --- Evaluation
def evaluate_clustering(X, y_true, model_name, model_obj):
    start_time = time.time()
    model_obj.fit(X)
    end_time = time.time()
    runtime = end_time - start_time

    if hasattr(model_obj, "predict"):
        y_pred = model_obj.predict(X)
    else:
        y_pred = model_obj.labels_

    metrics = {
        "Model": model_name,
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "Silhouette": silhouette_score(X, y_pred),
        "Weighted Gower": compute_weighted_gower_score(X, y_pred),
    }

    return metrics

# --- Highlighting
def highlight_best(df, metrics=["ARI", "NMI", "Silhouette", "Weighted Gower"]):
    df_display = df.copy()

    for metric in metrics:
        if metric in df_display.columns:
            maximize = metric not in ["Weighted Gower"]  # Lower Gower is better
            best_value = df_display[metric].max() if maximize else df_display[metric].min()

            def format_func(x):
                if pd.isna(x):
                    return "-"
                elif np.isclose(x, best_value, atol=1e-6):
                    return f"**{x:.4f}**"
                else:
                    return f"{x:.4f}"

            df_display[metric] = df_display[metric].apply(format_func)

    if "Confidence" in df_display.columns:
        df_display["Confidence"] = df_display["Confidence"].apply(
            lambda x: "-" if pd.isna(x) else f"{x:.4f}"
        )

    return df_display
