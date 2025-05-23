# tests/test_vfkm.py

import pytest
import numpy as np
from model import RobustEntropyVFKM

def test_empty_input():
    model = RobustEntropyVFKM(n_clusters=3)
    X_empty = np.empty((0, 2))
    with pytest.raises(ValueError):
        model.fit(X_empty)

def test_single_point_input():
    model = RobustEntropyVFKM(n_clusters=1)  # Changed from 3 -> 1
    X_single = np.array([[0.5, 1.5]])
    model.fit(X_single)
    assert model.u.shape == (1, 1)
    assert np.isclose(model.u.sum(), 1.0)

def test_more_clusters_than_points():
    model = RobustEntropyVFKM(n_clusters=20)
    X = np.random.randn(5, 2)
    # Since we know KMeans will fail, expect ValueError
    with pytest.raises(ValueError):
        model.fit(X)


def test_predict_before_fit():
    model = RobustEntropyVFKM(n_clusters=2)
    X_new = np.random.randn(5, 2)
    with pytest.raises(ValueError):
        model.predict(X_new)

def test_predict_proba_before_fit():
    model = RobustEntropyVFKM(n_clusters=2)
    X_new = np.random.randn(5, 2)
    with pytest.raises(ValueError):
        model.predict_proba(X_new)

def test_lambda_small_no_nan():
    model = RobustEntropyVFKM(n_clusters=3, lambda_entropy=1e-5)
    X = np.random.randn(10, 5)
    model.fit(X)
    assert not np.isnan(model.u).any()


def test_mahalanobis_edge_case():
    model = RobustEntropyVFKM(n_clusters=3, use_mahalanobis=True)
    X = np.random.randn(10, 1)  # 1D data
    model.fit(X)
    assert model.u.shape == (10, 3)
    assert np.allclose(model.u.sum(axis=1), 1.0, atol=1e-5)

def test_dynamic_lambda_adaptation():
    model = RobustEntropyVFKM(n_clusters=4, use_dynamic_lambda=True, lambda_adapt_rate=0.5)
    X = np.random.randn(50, 10)
    model.fit(X)
    assert model.lambda_entropy > 0.05  # Should respect minimum lambda

def test_one_cluster_case():
    model = RobustEntropyVFKM(n_clusters=1)
    X = np.random.randn(30, 5)
    model.fit(X)
    assert np.allclose(model.u[:, 0], 1.0, atol=1e-5)
    assert np.allclose(model.u.sum(axis=1), 1.0, atol=1e-5)

def test_predict_and_proba_shapes():
    model = RobustEntropyVFKM(n_clusters=5)
    X = np.random.randn(20, 4)
    model.fit(X)

    preds = model.predict(X)
    probas = model.predict_proba(X)

    assert preds.shape == (20,)
    assert probas.shape == (20, 5)
    assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-5)

def test_mahalanobis_vs_euclidean_behavior():
    """
    Mahalanobis and Euclidean distances should behave differently
    when feature scales are different.
    """
    X = np.random.randn(100, 3) * np.array([1.0, 10.0, 100.0])  # Different scales

    model_euc = RobustEntropyVFKM(n_clusters=3, use_mahalanobis=False)
    model_euc.fit(X)

    model_mah = RobustEntropyVFKM(n_clusters=3, use_mahalanobis=True)
    model_mah.fit(X)

    # The assignments should differ
    euc_labels = model_euc.labels_
    mah_labels = model_mah.labels_

    assert not np.all(euc_labels == mah_labels), "Mahalanobis and Euclidean produced identical clusters unexpectedly."

def test_mahalanobis_identity_covariance_reduces_to_euclidean():
    """
    If the covariance matrix is close to identity, 
    Mahalanobis distance should behave like Euclidean.
    """
    X = np.random.randn(100, 5)  # Standard normal -> Identity covariance roughly

    # Force Identity covariance artificially
    model_mah = RobustEntropyVFKM(n_clusters=3, use_mahalanobis=True)
    model_mah.inv_cov = np.eye(5)  # Manually override

    dists_mahal = model_mah._compute_distances(X, np.zeros((3, 5)), model_mah.inv_cov)
    dists_euc = np.linalg.norm(X[:, None, :] - np.zeros((1, 3, 5)), axis=2) ** 2

    assert np.allclose(dists_mahal, dists_euc, atol=1e-4), "Mahalanobis with Identity covariance should match Euclidean."

