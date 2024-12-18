import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class StatisticalThresholdClassifier(BaseEstimator, ClassifierMixin):
    """
    A simple statistical classifier that uses mean and std of features for classification.
    Supports multiple decision rules and thresholding strategies.
    """

    def __init__(self, strategy='mahalanobis', threshold_factor=1.0, use_window_stats=False):
        """
        Initialize the classifier.

        Args:
            strategy: Decision strategy ('mahalanobis', 'zscore', or 'combined')
            threshold_factor: Factor to adjust threshold sensitivity
            use_window_stats: Whether to compute additional window-level statistics
                            (not needed if differences are pre-computed per window)
        """
        self.strategy = strategy
        self.threshold_factor = threshold_factor
        self.class_stats = {}
        self.feature_names = ['pupil_size_diff', 'fixation_duration_diff']
        self.use_window_stats = use_window_stats  # Usually False since diffs are pre-computed

    def _compute_window_stats(self, X_window):
        """
        Compute window-level statistics (rarely needed since diffs are usually pre-computed).
        This is only used if use_window_stats=True.

        Args:
            X_window: Window of shape [window_size, n_features]

        Returns:
            Array of statistics for the window
        """
        # NOTE: This is rarely needed since differences are usually pre-computed per window
        stats = []
        for i in range(X_window.shape[1]):
            feature_values = X_window[:, i]
            diff = np.max(feature_values) - np.min(feature_values)
            stats.extend([diff])
        return np.array(stats)

    def fit(self, X, y):
        """
        Compute mean and std for each class.

        Args:
            X: If use_window_stats=False (default): Features array of shape [n_windows, n_features]
               where features are pre-computed differences
               If use_window_stats=True: Array of shape [n_windows, window_size, n_features]
            y: Labels array of shape [n_samples]
        """
        # If using window stats, compute them first (rarely needed)
        if self.use_window_stats and len(X.shape) == 3:
            X = np.array([self._compute_window_stats(window) for window in X])
        self.classes_ = np.unique(y)

        # Compute statistics for each class
        for class_label in self.classes_:
            class_mask = (y == class_label)
            class_data = X[class_mask]

            self.class_stats[class_label] = {
                'mean': np.mean(class_data, axis=0),
                'std': np.std(class_data, axis=0),
                'cov': np.cov(class_data.T) if len(class_data) > 1 else np.eye(X.shape[1])
            }

        return self

    def _mahalanobis_distance(self, X, class_label):
        """Calculate Mahalanobis distance to class centroid."""
        mean = self.class_stats[class_label]['mean']
        cov = self.class_stats[class_label]['cov']
        cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))  # Add small regularization

        diff = X - mean
        return np.sqrt(np.sum(np.dot(diff, cov_inv) * diff, axis=1))

    def _zscore_distance(self, X, class_label):
        """Calculate normalized z-score distance to class mean."""
        mean = self.class_stats[class_label]['mean']
        std = self.class_stats[class_label]['std']
        return np.mean(np.abs((X - mean) / (std + 1e-6)), axis=1)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X: If use_window_stats=False (default): Features array of shape [n_windows, n_features]
               where features are pre-computed differences
               If use_window_stats=True: Array of shape [n_windows, window_size, n_features]

        Returns:
            Array of predicted labels
        """
        if self.strategy == 'mahalanobis':
            # Use Mahalanobis distance to class centroids
            distances = np.array([
                self._mahalanobis_distance(X, class_label)
                for class_label in self.classes_
            ]).T

        elif self.strategy == 'zscore':
            # Use mean z-score distance
            distances = np.array([
                self._zscore_distance(X, class_label)
                for class_label in self.classes_
            ]).T

        else:  # combined
            # Combine both distances with equal weight
            mahalanobis_dist = np.array([
                self._mahalanobis_distance(X, class_label)
                for class_label in self.classes_
            ]).T

            zscore_dist = np.array([
                self._zscore_distance(X, class_label)
                for class_label in self.classes_
            ]).T

            # Normalize each distance metric
            mahalanobis_dist = mahalanobis_dist / np.max(mahalanobis_dist)
            zscore_dist = zscore_dist / np.max(zscore_dist)

            distances = (mahalanobis_dist + zscore_dist) / 2

        # Apply threshold factor to favor minority class (class 1)
        distances[:, 1] = distances[:, 1] / self.threshold_factor

        return self.classes_[np.argmin(distances, axis=1)]

    def evaluate(self, X, y_true):
        """
        Evaluate classifier performance.

        Args:
            X: Features array
            y_true: True labels

        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

        # Print class statistics
        print("\nClass Statistics:")
        for class_label in self.classes_:
            print(f"\nClass {class_label}:")
            print(f"Mean: {self.class_stats[class_label]['mean']}")
            print(f"Std: {self.class_stats[class_label]['std']}")

        # Print evaluation metrics
        print("\nEvaluation Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print("\nConfusion Matrix:")
        print(metrics['confusion_matrix'])

        return metrics


# Example usage:
def evaluate_baseline(X_train, y_train, X_test, y_test):
    """
    Evaluate the statistical classifier with different strategies.
    """
    strategies = ['mahalanobis', 'zscore', 'combined']
    threshold_factors = [1.0, 1.5, 2.0]  # Different sensitivity levels

    results = {}

    for strategy in strategies:
        strategy_results = {}

        for threshold in threshold_factors:
            print(f"\nTesting {strategy} strategy with threshold factor {threshold}")
            clf = StatisticalThresholdClassifier(
                strategy=strategy,
                threshold_factor=threshold
            )

            clf.fit(X_train, y_train)
            metrics = clf.evaluate(X_test, y_test)
            strategy_results[threshold] = metrics

        results[strategy] = strategy_results

    return results


def analyze_dynamic_windows(X_train, Y_train, X_test, Y_test, X_val=None, Y_val=None):
    """
    Analyze dynamic windows using the statistical classifier.

    Args:
        X_train: Training windows from create_dynamic_time_series
        Y_train: Training labels
        X_test: Test windows
        Y_test: Test labels
        X_val: Optional validation windows
        Y_val: Optional validation labels
    """
    # Extract the difference features from windows
    # The last two columns are the max differences we computed in create_dynamic_time_series
    n_base_features = X_train.shape[2] - 2  # Original features minus the two difference features

    # Extract just the difference features for each window
    train_diffs = X_train[:, -1, n_base_features:]  # Take last timestep, only diff features
    test_diffs = X_test[:, -1, n_base_features:]

    if X_val is not None:
        val_diffs = X_val[:, -1, n_base_features:]

    print("\nData Shapes:")
    print(f"Training differences shape: {train_diffs.shape}")
    print(f"Test differences shape: {test_diffs.shape}")
    if X_val is not None:
        print(f"Validation differences shape: {val_diffs.shape}")

    # Create and evaluate classifier with different strategies
    strategies = ['mahalanobis', 'zscore', 'combined']
    threshold_factors = [1.0, 1.5, 2.0, 2.5, 3.0]

    best_f1 = 0
    best_config = None
    best_clf = None

    results = {}

    print("\nEvaluating different classifier configurations...")
    for strategy in strategies:
        strategy_results = {}
        for threshold in threshold_factors:
            print(f"\nTesting {strategy} strategy with threshold factor {threshold}")

            clf = StatisticalThresholdClassifier(
                strategy=strategy,
                threshold_factor=threshold
            )

            # Fit on training data
            clf.fit(train_diffs, Y_train)

            # Evaluate on validation set if provided
            if X_val is not None:
                metrics = clf.evaluate(val_diffs, Y_val)
            else:
                metrics = clf.evaluate(test_diffs, Y_test)

            strategy_results[threshold] = metrics

            # Track best configuration based on F1 score
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_config = {'strategy': strategy, 'threshold': threshold}
                best_clf = clf

        results[strategy] = strategy_results

    print("\nBest Configuration:")
    print(f"Strategy: {best_config['strategy']}")
    print(f"Threshold Factor: {best_config['threshold']}")
    print(f"Best F1 Score: {best_f1:.4f}")

    # Final evaluation on test set if validation was used
    if X_val is not None:
        print("\nFinal Evaluation on Test Set:")
        best_clf.evaluate(test_diffs, Y_test)

    return results, best_clf


# Usage example right after create_dynamic_time_series calls:
