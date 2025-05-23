import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

max_iter = 100

class BaselineKMeansModel:
    """
    Baseline KMeans clustering model for benchmarking.

    Args:
        n_clusters (int): Number of clusters
        random_state (int): Random seed
    """

    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X):
        """
        Fit the KMeans model.

        Args:
            X (np.ndarray): Data matrix (n_samples, n_features)
        """
        self.model = KMeans(n_clusters=self.n_clusters, n_init=10, max_iter = max_iter, random_state=self.random_state)
        self.labels_ = self.model.fit_predict(X)

    def predict(self, X):
        """
        Predict cluster labels for X.

        Args:
            X (np.ndarray): Data matrix
        Returns:
            np.ndarray: Cluster labels
        """
        return self.model.predict(X)
    
class BaselineGMMModel:
    """
    Baseline GMM clustering model for benchmarking.

    Args:
        n_clusters (int): Number of clusters
        random_state (int): Random seed
    """

    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X):
        """
        Fit the GMM model.

        Args:
            X (np.ndarray): Data matrix (n_samples, n_features)
        """
        self.model = GaussianMixture(n_components=self.n_clusters, random_state=self.random_state)
        self.labels_ = self.model.fit_predict(X)

    def predict(self, X):
        """
        Predict cluster labels for X.

        Args:
            X (np.ndarray): Data matrix
        Returns:
            np.ndarray: Cluster labels
        """
        return self.model.predict(X)
    
class SoftKMeans:
    """
    Simple soft KMeans with temperature control (no entropy penalty).
    """
    def __init__(self, n_clusters=10, temperature=1.0, max_iter=max_iter, tol=1e-4):
        self.n_clusters = n_clusters
        self.temperature = temperature
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape
        kmeans = KMeans(n_clusters=self.n_clusters, max_iter = self.max_iter, n_init=10, random_state=42)
        kmeans.fit(X)
        self.centroids = kmeans.cluster_centers_

        for iteration in range(self.max_iter):
            dists = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2) ** 2
            logits = -dists / self.temperature
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            self.u = probs / probs.sum(axis=1, keepdims=True)

            new_centroids = (self.u.T @ X) / (self.u.sum(axis=0)[:, None] + 1e-8)

            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

        self.labels_ = self.u.argmax(axis=1)

    def predict(self, X_new):
        dists = np.linalg.norm(X_new[:, None, :] - self.centroids[None, :, :], axis=2) ** 2
        logits = -dists / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        u_pred = probs / probs.sum(axis=1, keepdims=True)
        return u_pred.argmax(axis=1)
    
    def predict_proba(self, X_new):
        """
        Predict soft membership probabilities for new data points.
        
        Args:
            X_new (np.ndarray): New data matrix (n_samples, n_features)

        Returns:
            np.ndarray: Membership matrix (n_samples, n_clusters)
        """
        dists = np.linalg.norm(X_new[:, None, :] - self.centroids[None, :, :], axis=2) ** 2
        logits = -dists / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        u_pred = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return u_pred

    
class AnnealedSoftKMeans:
    """
    Soft KMeans with temperature annealing over iterations.
    """

    def __init__(self, n_clusters=10, init_temp=5.0, final_temp=0.5, max_iter=max_iter, tol=1e-4):
        self.n_clusters = n_clusters
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape
        kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, n_init=10, random_state=42)
        kmeans.fit(X)
        self.centroids = kmeans.cluster_centers_

        temps = np.linspace(self.init_temp, self.final_temp, self.max_iter)

        for iteration, temp in enumerate(temps):
            dists = np.linalg.norm(X[:, None, :] - self.centroids[None, :, :], axis=2) ** 2
            logits = -dists / temp
            logits -= logits.max(axis=1, keepdims=True)
            probs = np.exp(logits)
            self.u = probs / probs.sum(axis=1, keepdims=True)

            new_centroids = (self.u.T @ X) / (self.u.sum(axis=0)[:, None] + 1e-8)

            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break

            self.centroids = new_centroids

        self.labels_ = self.u.argmax(axis=1)

    def predict(self, X_new):
        final_temp = self.final_temp
        dists = np.linalg.norm(X_new[:, None, :] - self.centroids[None, :, :], axis=2) ** 2
        logits = -dists / final_temp
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        u_pred = probs / probs.sum(axis=1, keepdims=True)
        return u_pred.argmax(axis=1)
    
    def predict_proba(self, X_new):
        """
        Predict soft membership probabilities for new data points.
        
        Args:
            X_new (np.ndarray): New data matrix (n_samples, n_features)

        Returns:
            np.ndarray: Membership matrix (n_samples, n_clusters)
        """
        dists = np.linalg.norm(X_new[:, None, :] - self.centroids[None, :, :], axis=2) ** 2
        logits = -dists / self.final_temp
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        u_pred = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return u_pred



class RobustEntropyVFKMOld:
    """
    Entropy-Regularized Variational Fuzzy K-Means 
    with optional KL regularization, dynamic lambda adaptation, 
    and Mahalanobis distance.
    """

    def __init__(self, n_clusters=10, 
                 lambda_entropy=5.0, 
                 lambda_kl=0.0, 
                 anneal_gamma=0.0,
                 lambda_adapt_rate=0.0,
                 lambda_min_value=0.05,
                 use_dynamic_lambda=False,
                 use_mahalanobis=False,
                 max_iter=max_iter, tol=1e-4, verbose=False):
        """
        Args:
            use_mahalanobis (bool): Whether to use Mahalanobis distance instead of Euclidean.
        """
        self.n_clusters = n_clusters
        self.lambda_entropy = lambda_entropy
        self.lambda_kl = lambda_kl
        self.anneal_gamma = anneal_gamma
        self.lambda_adapt_rate = lambda_adapt_rate
        self.lambda_min_value = lambda_min_value
        self.use_dynamic_lambda = use_dynamic_lambda
        self.use_mahalanobis = use_mahalanobis
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def _compute_distances(self, X, centroids, inv_cov=None):
        """
        Compute either Mahalanobis or Euclidean distances.
        """
        if self.use_mahalanobis:
            diffs = X[:, None, :] - centroids[None, :, :]
            dists = np.einsum('nik,kl,nil->ni', diffs, inv_cov, diffs)
        else:
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2) ** 2
        return dists

    def fit(self, X):
        """
        Fit the model to data.
        """
        n_samples, n_features = X.shape
        self.X = X

        if self.use_mahalanobis:
            cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(n_features)  # Regularization
            self.inv_cov = np.linalg.inv(cov)
        else:
            self.inv_cov = None

        # Warm-start centroids
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        kmeans.fit(X)
        self.centroids = kmeans.cluster_centers_

        # Initialize memberships
        dists = self._compute_distances(X, self.centroids, self.inv_cov)
        logits = -dists / self.lambda_entropy
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        self.u = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        if self.verbose:
            print(f"[Init] Warm-started using KMeans")
            print(f"[Init] Max logits: {logits.max():.2f}, Min logits: {logits.min():.2f}")

        loss_prev = None
        self.loss_history = []  # Track convergence

        self.u_prev = self.u.copy()  # Initialize for KL term

        for iteration in range(self.max_iter):
            u_old = self.u.copy()

            # Update centroids
            self.centroids = (self.u.T @ X) / (self.u.sum(axis=0)[:, None] + 1e-8)

            # Update memberships
            dists = self._compute_distances(X, self.centroids, self.inv_cov)

            # Revert back to the original update rule
            logits = -dists / self.lambda_entropy
            if self.lambda_kl > 0.0:
                logits += (self.lambda_kl / self.lambda_entropy) * np.log(self.u_prev + 1e-10)


            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            self.u = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # Compute Free Energy Loss
            recon_loss = np.sum(self.u * dists)

            # 🔽 Corrected entropy term
            entropy_loss = -self.lambda_entropy * np.sum(self.u * np.log(self.u + 1e-10))

            # KL anchoring term remains correct
            kl_loss = 0.0
            if self.lambda_kl > 0.0:
                kl_loss = self.lambda_kl * np.sum(self.u * (np.log(self.u + 1e-10) - np.log(self.u_prev + 1e-10)))


            total_loss = recon_loss + entropy_loss + kl_loss
            self.loss_history.append(total_loss)

            # Dynamic lambda adaptation
            if self.use_dynamic_lambda and loss_prev is not None:
                delta_loss = (total_loss - loss_prev) / (abs(loss_prev) + 1e-8)
                self.lambda_entropy *= (1 + self.lambda_adapt_rate * delta_loss)
                self.lambda_entropy = max(self.lambda_entropy, self.lambda_min_value)

            loss_prev = total_loss

            # Annealing decay
            if self.anneal_gamma > 0:
                self.lambda_entropy *= np.exp(-self.anneal_gamma)

            # Convergence Check
            diff = np.linalg.norm(self.u - u_old)
            if self.verbose and (iteration % 50 == 0 or iteration == self.max_iter - 1):
                print(f"[Iter {iteration}] Total Loss: {total_loss:.4f} | u diff: {diff:.6f} | λ_entropy: {self.lambda_entropy:.6f}")

            if diff < self.tol:
                if self.verbose:
                    print(f"[Converged] Iteration {iteration}")
                break

            self.u_prev = self.u.copy()  # Update for next KL term

        self.labels_ = self.u.argmax(axis=1)
        return self


    def predict(self, X_new):
        """
        Predict cluster assignments for new data points.
        """
        if not hasattr(self, "centroids"):
            raise ValueError("Model must be fitted before prediction.")

        dists = self._compute_distances(X_new, self.centroids, self.inv_cov)
        logits = -dists / self.lambda_entropy
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        u_pred = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return u_pred.argmax(axis=1)

    def predict_proba(self, X_new):
        """
        Predict soft membership probabilities for new data points.
        """
        if not hasattr(self, "centroids"):
            raise ValueError("Model must be fitted before prediction.")

        dists = self._compute_distances(X_new, self.centroids, self.inv_cov)
        logits = -dists / self.lambda_entropy
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        u_pred = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return u_pred


class RobustEntropyVFKM:
    """
    Entropy-Regularized Variational Fuzzy K-Means
    with optional KL regularization, dynamic lambda adaptation,
    and Mahalanobis distance.
    """

    def __init__(self, n_clusters=10,
                 lambda_entropy=5.0,
                 lambda_kl=0.0,
                 anneal_gamma=0.0,
                 lambda_adapt_rate=0.0,
                 lambda_min_value=0.05,
                 use_dynamic_lambda=False,
                 use_mahalanobis=False,
                 max_iter=100, tol=1e-4, verbose=False):
        """
        Args:
            use_mahalanobis (bool): Whether to use Mahalanobis distance instead of Euclidean.
        """
        self.n_clusters = n_clusters
        self.lambda_entropy = lambda_entropy
        self.lambda_kl = lambda_kl
        self.anneal_gamma = anneal_gamma
        self.lambda_adapt_rate = lambda_adapt_rate
        self.lambda_min_value = lambda_min_value
        self.use_dynamic_lambda = use_dynamic_lambda
        self.use_mahalanobis = use_mahalanobis
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def _compute_distances(self, X, centroids, inv_cov=None):
        """
        Compute either Mahalanobis or Euclidean distances.
        """
        if self.use_mahalanobis:
            diffs = X[:, None, :] - centroids[None, :, :]
            # Using einsum for efficient batch Mahalanobis distance
            # (N, K, D) * (D, D) * (N, K, D) -> (N, K)
            dists = np.einsum('nik,kl,nil->ni', diffs, inv_cov, diffs)
        else:
            dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2) ** 2
        return dists

    def fit(self, X):
        """
        Fit the model to data.
        """
        n_samples, n_features = X.shape
        self.X = X

        if self.use_mahalanobis:
            # Add small regularization to covariance for numerical stability
            cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(n_features)
            self.inv_cov = np.linalg.inv(cov)
        else:
            self.inv_cov = None

        # Warm-start centroids using KMeans for better initialization
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        kmeans.fit(X)
        self.centroids = kmeans.cluster_centers_

        # Initialize memberships using the initial centroids and the correct update rule
        # This initial step also needs to use the correct effective_lambda
        effective_lambda_init = self.lambda_entropy - self.lambda_kl
        if effective_lambda_init <= 1e-8: # Ensure positive and non-zero for stability
            effective_lambda_init = 1e-8

        dists_init = self._compute_distances(X, self.centroids, self.inv_cov)
        logits_init = -dists_init / effective_lambda_init
        # No u_prev for initial calculation, so KL term is 0 here.
        logits_init -= logits_init.max(axis=1, keepdims=True) # For numerical stability before exp
        exp_logits_init = np.exp(logits_init)
        self.u = exp_logits_init / exp_logits_init.sum(axis=1, keepdims=True)

        if self.verbose:
            print(f"[Init] Warm-started using KMeans")
            print(f"[Init] Max logits: {logits_init.max():.2f}, Min logits: {logits_init.min():.2f}")

        loss_prev = None
        self.loss_history = [] # Track convergence

        self.u_prev = self.u.copy()  # Initialize for KL term in subsequent iterations

        for iteration in range(self.max_iter):
            u_old = self.u.copy()

            # Update centroids (M-step)
            self.centroids = (self.u.T @ X) / (self.u.sum(axis=0)[:, None] + 1e-8)

            # Compute distances to updated centroids
            dists = self._compute_distances(X, self.centroids, self.inv_cov)

            # Calculate effective lambda for the current iteration
            effective_lambda = self.lambda_entropy - self.lambda_kl
            if effective_lambda <= 1e-8: # Ensure positive and non-zero for stability
                # If lambda_entropy <= lambda_kl, the "temperature" becomes non-positive.
                # This changes the nature of the assignments significantly (e.g., hard assignments if 0).
                # For strict adherence to the derived formula's interpretation, we assume effective_lambda > 0.
                # Setting a small positive value to prevent division by zero.
                effective_lambda = 1e-8

            # Calculate logits based on the derived formula
            # Term 1: -||x_i - mu_k||^2 / (lambda_entropy - lambda_kl)
            logits = -dists / effective_lambda

            # Term 2: - (lambda_kl * log(u_ik^prev)) / (lambda_entropy - lambda_kl)
            if self.lambda_kl > 0.0:
                # Note the negative sign as per the derivation
                logits -= (self.lambda_kl / effective_lambda) * np.log(self.u_prev + 1e-10)

            # Normalize logits for numerical stability before exponentiation
            logits -= logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits)
            self.u = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # Compute Free Energy Loss (for tracking convergence)
            recon_loss = np.sum(self.u * dists)
            entropy_loss = -self.lambda_entropy * np.sum(self.u * np.log(self.u + 1e-10))
            kl_loss = 0.0
            if self.lambda_kl > 0.0:
                kl_loss = self.lambda_kl * np.sum(self.u * (np.log(self.u + 1e-10) - np.log(self.u_prev + 1e-10)))
            total_loss = recon_loss + entropy_loss + kl_loss
            self.loss_history.append(total_loss)

            # Dynamic lambda adaptation (if enabled)
            if self.use_dynamic_lambda and loss_prev is not None:
                delta_loss = (total_loss - loss_prev) / (abs(loss_prev) + 1e-8)
                self.lambda_entropy *= (1 + self.lambda_adapt_rate * delta_loss)
                self.lambda_entropy = max(self.lambda_entropy, self.lambda_min_value)

            loss_prev = total_loss

            # Annealing decay (if enabled)
            if self.anneal_gamma > 0:
                self.lambda_entropy *= np.exp(-self.anneal_gamma)

            # Convergence Check
            diff = np.linalg.norm(self.u - u_old)
            if self.verbose and (iteration % 50 == 0 or iteration == self.max_iter - 1):
                print(f"[Iter {iteration}] Total Loss: {total_loss:.4f} | u diff: {diff:.6f} | λ_entropy: {self.lambda_entropy:.6f}")

            if diff < self.tol:
                if self.verbose:
                    print(f"[Converged] Iteration {iteration}")
                break

            self.u_prev = self.u.copy()  # Update for next KL term

        self.labels_ = self.u.argmax(axis=1)
        return self


    def predict(self, X_new):
        """
        Predict cluster assignments for new data points.
        """
        if not hasattr(self, "centroids"):
            raise ValueError("Model must be fitted before prediction.")

        # Use the final lambda_entropy value from fitting
        effective_lambda_pred = self.lambda_entropy - self.lambda_kl
        if effective_lambda_pred <= 1e-8:
            effective_lambda_pred = 1e-8

        dists = self._compute_distances(X_new, self.centroids, self.inv_cov)
        logits = -dists / effective_lambda_pred
        # For prediction, u_prev is not directly involved in the same way as training
        # The prediction is based on the final learned centroids and parameters.
        # If u_prev is needed for a specific prediction logic, it would need to be passed or handled.
        # Assuming standard prediction where only final centroids and parameters determine assignment.
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        u_pred = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return u_pred.argmax(axis=1)

    def predict_proba(self, X_new):
        """
        Predict soft membership probabilities for new data points.
        """
        if not hasattr(self, "centroids"):
            raise ValueError("Model must be fitted before prediction.")

        # Use the final lambda_entropy value from fitting
        effective_lambda_pred = self.lambda_entropy - self.lambda_kl
        if effective_lambda_pred <= 1e-8:
            effective_lambda_pred = 1e-8

        dists = self._compute_distances(X_new, self.centroids, self.inv_cov)
        logits = -dists / effective_lambda_pred
        # For prediction, u_prev is not directly involved in the same way as training
        logits -= logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        u_pred = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        return u_pred


