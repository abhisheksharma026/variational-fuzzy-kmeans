# src/runner.py

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedKFold
from model import BaselineKMeansModel, BaselineGMMModel, RobustEntropyVFKM, SoftKMeans, AnnealedSoftKMeans
from utils import (
    load_digits_data,
    load_breast_cancer_data,
    load_usps_data,
    load_mnist_data,
    highlight_best,
    evaluate_clustering,
)

# --- Benchmark
def run_benchmark(n_splits=5):
    dataset_loaders = {
        "Breast Cancer": load_breast_cancer_data,
        "Digits": load_digits_data,
        "USPS": load_usps_data,
        "MNIST": load_mnist_data
    }

    all_results = []

    for dataset_name, loader_fn in dataset_loaders.items():
        print(f"\n🔍 Running 5-Fold Cross-Validation on {dataset_name}...")

        X, y_true = loader_fn()
        n_clusters = len(np.unique(y_true))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_true)):
            X_fold = X[test_idx]
            y_fold = y_true[test_idx]

            models = {
                    "KMeans": BaselineKMeansModel(n_clusters=n_clusters, random_state=fold_idx),
                    "GMM": BaselineGMMModel(n_clusters=n_clusters, random_state=fold_idx),
                    "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
                    "Soft-KMeans": SoftKMeans(n_clusters=n_clusters, temperature=1.0),
                    "Annealed-Soft-KMeans": AnnealedSoftKMeans(n_clusters=n_clusters, init_temp=5.0, final_temp=0.5),
                    # Main Model
                    "Variational-BFKM": RobustEntropyVFKM(n_clusters=n_clusters, lambda_entropy=5.0, lambda_kl=0.5, anneal_gamma=0.02, verbose=False),
                    # Ablations
                    "Variational-BFKM (No Entropy)": RobustEntropyVFKM(n_clusters=n_clusters, lambda_entropy=1e-5, lambda_kl=0.5, verbose=False),
                    "Variational-BFKM (No KL)": RobustEntropyVFKM(n_clusters=n_clusters, lambda_entropy=5.0, lambda_kl=0.0, verbose=False),
                    "Variational-BFKM (No Anneal)": RobustEntropyVFKM(n_clusters=n_clusters, lambda_entropy=5.0, lambda_kl=0.5, anneal_gamma=0.0, verbose=False),
                    "Variational-BFKM (No Entropy + No KL)": RobustEntropyVFKM(n_clusters=n_clusters, lambda_entropy=1e-5, lambda_kl=0.0, verbose=False),
                    }

            for model_name, model_obj in models.items():
                metrics = evaluate_clustering(X_fold, y_fold, model_name, model_obj)
                metrics["Model"] = model_name
                fold_results.append(metrics)

        model_order = [
            "KMeans",
            "GMM",
            "Agglomerative",
            "Soft-KMeans",
            "Annealed-Soft-KMeans",
            "Variational-BFKM (No Entropy)",
            "Variational-BFKM (No KL)",
            "Variational-BFKM (No Anneal)",
            "Variational-BFKM (No Entropy + No KL)",
            "Variational-BFKM",
        ]

        df_fold = pd.DataFrame(fold_results)

        # Summary: Mean over folds (Runtime: sum)
        df_summary = df_fold.groupby("Model").agg({
            "ARI": "mean",
            "NMI": "mean",
            "Silhouette": "mean",
            "Weighted Gower": "mean",
        })

        df_summary = df_summary.reindex(model_order)
        df_summary["Dataset"] = dataset_name

        print(f"\n CV Benchmark Results for {dataset_name}:")
        print(highlight_best(df_summary).to_markdown(tablefmt="github"))

        all_results.append(df_summary)

    df_all = pd.concat(all_results)
    df_all.to_csv("benchmark_results_cv.csv")

if __name__ == "__main__":
    run_benchmark()
