import numpy as np

class WeightedKMeans:
    """
    A simple implementation of K-Means clustering algorithm with weighted features.
    """
    def __init__(self, n_clusters=8, max_iter=100, random_state=None):
        """Initialize the WeightedKMeans instance.

        Args:
            n_clusters (int, optional): Nunmer of clusters. Defaults to 8.
            max_iter (int, optional): Maximum number of iterations. Defaults to 100.
            random_state (int, optional): Random seed, used for reproducibility. Defaults to None.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X, weights):
        np.random.seed(self.random_state)
        # Initialize cluster centers randomly
        idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.cluster_centers_ = X[idx]

        for _ in range(self.max_iter):
            # Assign labels based on closest cluster center
            self.labels_ = np.argmin(
                np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2), axis=1
            )

            for i in range(self.n_clusters):
                cluster_points = X[self.labels_ == i]
                if len(cluster_points) == 0:
                    new_center_idx = np.random.randint(X.shape[0])
                    self.labels_[new_center_idx] = i
            # Update cluster centers based on the mean of the points in each cluster

            Y = X * weights
            new_centers = np.array(
                [Y[self.labels_ == i].mean(axis=0) for i in range(self.n_clusters)]
            )

            # Check for convergence
            if np.allclose(self.cluster_centers_, new_centers):
                break

            self.cluster_centers_ = np.array(new_centers)

            distances = np.linalg.norm(X - self.cluster_centers_[self.labels_], axis=1)
            self.inertia_ = np.sum(distances**2)

    def predict(self, X):
        """Predict cluster labels for new data points.

        Args:
            X (DataFrame): Dataset to predict.

        Returns:
            DataFrame: cluster labels for each data point.
        """
        return np.argmin(
            np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2), axis=1
        )
