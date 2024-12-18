import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import IsolationForest
import numpy as np
from sklearn.model_selection import cross_val_score
from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support, classification_report, confusion_matrix
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import numpy as np

import numpy as np
import torch
import faiss
import faiss.contrib.torch_utils
from sklearn.decomposition import PCA


class FAISSKNNClassifier:
    """
    Fast KNN classifier using FAISS for similarity search.
    Supports both GPU and CPU operations with automatic handling of tensor types.
    """

    def __init__(self, n_neighbors=5, pca_components=0.99, metric='l2', use_gpu=True):
        """
        Initialize the FAISS KNN classifier.

        Args:
            n_neighbors: Number of neighbors for classification
            pca_components: Number of PCA components or fraction of variance to keep
            metric: Distance metric ('l2' or 'inner_product')
            use_gpu: Whether to use GPU acceleration
        """
        self.n_neighbors = n_neighbors
        self.pca_components = pca_components
        self.metric = metric
        self.use_gpu = use_gpu and torch.cuda.is_available()

        self.index = None
        self.pca = None
        self.y_train = None
        self.n_classes = None

    def _create_index(self, d):
        """Create appropriate FAISS index based on settings"""
        if self.metric == 'l2':
            index = faiss.IndexFlatL2(d)
        else:  # inner_product
            index = faiss.IndexFlatIP(d)

        if self.use_gpu:
            # Use multiple GPUs if available
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                print(f"Using {n_gpus} GPUs")
                index = faiss.index_cpu_to_all_gpus(index)
            else:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)

        return index

    def _preprocess_data(self, X, fit=False):
        """Preprocess data with PCA dimensionality reduction"""
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        if fit:
            # Initialize and fit PCA
            if isinstance(self.pca_components, float):
                self.pca = PCA(n_components=self.pca_components, random_state=42)
            else:
                self.pca = PCA(n_components=min(X.shape[1], self.pca_components),
                               random_state=42)
            X_reduced = self.pca.fit_transform(X)
            print(f"Reduced dimensions from {X.shape[1]} to {X_reduced.shape[1]}")
        else:
            # Apply existing PCA transformation
            X_reduced = self.pca.transform(X)

        return X_reduced.astype(np.float32)

    def fit(self, X, y):
        """
        Fit the KNN classifier

        Args:
            X: Training data (numpy array or torch tensor)
            y: Training labels (numpy array or torch tensor)
        """
        # Convert y to numpy if needed
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        self.y_train = y
        self.n_classes = len(np.unique(y))

        # Preprocess data
        print("Preprocessing training data...")
        X_processed = self._preprocess_data(X, fit=True)

        # Create and train index
        print("Creating FAISS index...")
        self.index = self._create_index(X_processed.shape[1])

        print("Adding training data to index...")
        self.index.add(X_processed)

        return self

    def _get_neighbors(self, X):
        """Find nearest neighbors and their distances"""
        # Preprocess query data
        X_processed = self._preprocess_data(X, fit=False)

        # Search index
        distances, indices = self.index.search(X_processed, self.n_neighbors)

        return distances, indices

    def predict_proba(self, X):
        distances, indices = self._get_neighbors(X)

        # Modify weights using both distance and sample weights
        weights = 1.0 / (distances + 1e-8)
        if hasattr(self, 'weights'):
            weights *= self.weights[indices]

        weights /= weights.sum(axis=1, keepdims=True)
        probas = np.zeros((len(X), self.n_classes))

        for i in range(len(X)):
            neighbor_labels = self.y_train[indices[i]]
            for j, label in enumerate(neighbor_labels):
                probas[i, label] += weights[i, j]

        return probas

    def predict(self, X):
        """
        Predict class labels for test points

        Args:
            X: Test data (numpy array or torch tensor)

        Returns:
            Predicted class labels
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)


class FastFAISSKNNClassifier(FAISSKNNClassifier):
    """
    Enhanced version of FAISS KNN classifier with majority class subsampling
    for better efficiency with imbalanced datasets.
    """

    def __init__(self, n_neighbors=5, pca_components=0.95, metric='l2',
                 use_gpu=True, majority_ratio=0.5):
        super().__init__(n_neighbors, pca_components, metric, use_gpu)
        self.majority_ratio = majority_ratio

    def _subsample_majority_class(self, X, y):
        """Subsample the majority class to reduce computational load"""
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()

        majority_mask = (y == 0)
        minority_mask = (y == 1)

        n_majority = majority_mask.sum()
        n_keep = int(n_majority * self.majority_ratio)

        # Get indices to keep
        majority_indices = np.where(majority_mask)[0]
        keep_indices = majority_indices[np.random.choice(len(majority_indices),
                                                         n_keep, replace=False)]
        minority_indices = np.where(minority_mask)[0]
        keep_indices = np.concatenate([keep_indices, minority_indices])

        if isinstance(X, torch.Tensor):
            return X[keep_indices], y[keep_indices]
        return X[keep_indices], y[keep_indices]

    def fit(self, X, y):
        """Fit with majority class subsampling"""
        print("Subsampling majority class...")
        X_sampled, y_sampled = self._subsample_majority_class(X, y)

        if hasattr(self, 'sample_weights'):
            # Convert sample weights to distance weights for FAISS
            self.weights = 1 / (1 + np.exp(-self.sample_weights))


        return super().fit(X_sampled, y_sampled)


class LatentSpaceClassifier:
    """
    A hybrid approach combining FAISS-based KNN, clustering, and anomaly detection
    for imbalanced classification in latent space.
    """

    def __init__(
            self,
            vae_model,
            n_neighbors=5,
            contamination=0.03,
            device='cuda'
    ):
        self.vae_model = vae_model
        self.device = device
        self.n_neighbors = n_neighbors
        self.contamination = contamination

        # Initialize sub-models
        self.knn = FastFAISSKNNClassifier(
            n_neighbors=n_neighbors,
            pca_components=0.99,
            metric='l2',
            use_gpu=(device == 'cuda'),
            majority_ratio=0.99
        )
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )

    def get_latent_features(self, loader, desc="Extracting features"):
        """Extract latent space features from data loader"""
        self.vae_model.eval()
        features = []
        labels = []

        with torch.no_grad():
            for x, y in tqdm(loader, desc=desc):
                x = x.to(self.device)
                mu, logvar = self.vae_model.encode(x)
                z = torch.cat([mu, logvar], dim=1)
                features.append(z)
                labels.append(y)

        features = torch.cat(features, dim=0)
        labels = torch.cat(labels, dim=0)

        return features, labels

    def fit(self, train_loader, val_loader=None):
        """
        Fit the hybrid model using both normal and anomaly detection approaches
        """
        print("\nStep 1/4: Extracting training features...")
        X_train, y_train = self.get_latent_features(train_loader, desc="Processing training data")

        # Move to CPU for isolation forest
        X_train_cpu = X_train.cpu().numpy()
        y_train_cpu = y_train.cpu().numpy()

        if val_loader:
            print("\nStep 2/4: Extracting validation features...")
            X_val, y_val = self.get_latent_features(val_loader, desc="Processing validation data")

        # Fit isolation forest for anomaly detection
        print("\nStep 3/4: Training Isolation Forest...")
        self.isolation_forest.fit(X_train_cpu)
        print("Isolation Forest training completed.")
        anomaly_scores = self.isolation_forest.score_samples(X_train_cpu)
        sample_weights = self._compute_sample_weights(y_train_cpu, anomaly_scores)
        self.sample_weights = sample_weights

        print("\nStep 4/4: Training FAISS KNN...")
        self.knn.fit(X_train, y_train)

        # Validate if validation set provided
        if val_loader:
            val_scores = self.evaluate(val_loader)
            print("\nValidation Metrics:")
            for metric, value in val_scores.items():
                print(f"{metric}: {value:.4f}")

        return self

    def _compute_sample_weights(self, y_train, anomaly_scores):
        """
        Compute sample weights based on class distribution and anomaly scores
        """
        # Class balance weights
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        base_weights = class_weights[y_train]

        # Normalize anomaly scores to [0,1]
        normalized_scores = (anomaly_scores - anomaly_scores.min()) / (
                anomaly_scores.max() - anomaly_scores.min())

        # Combine class weights with anomaly scores
        weights = base_weights * (1 + normalized_scores)

        return weights

    def predict_proba(self, loader):
        """
        Predict class probabilities using the hybrid approach
        """
        X_test, _ = self.get_latent_features(loader)

        # Get anomaly scores (on CPU)
        X_test_cpu = X_test.cpu().numpy()
        anomaly_scores = self.isolation_forest.score_samples(X_test_cpu)

        # Get KNN probabilities using FAISS
        knn_probs = self.knn.predict_proba(X_test)

        # Adjust probabilities based on anomaly scores
        adjusted_probs = self._adjust_probabilities(knn_probs, anomaly_scores)

        return adjusted_probs

    def _adjust_probabilities(self, base_probs, anomaly_scores):
        """
        Adjust predictions based on anomaly scores
        """
        # Normalize anomaly scores to [0,1]
        normalized_scores = (anomaly_scores - anomaly_scores.min()) / (
                anomaly_scores.max() - anomaly_scores.min())

        # Adjust minority class probability based on anomaly score
        adjusted_probs = base_probs.copy()
        adjusted_probs[:, 1] *= (1 - normalized_scores)  # Reduce minority prob for anomalies

        # Renormalize
        row_sums = adjusted_probs.sum(axis=1)
        adjusted_probs = adjusted_probs / row_sums[:, np.newaxis]

        return adjusted_probs

    def predict(self, loader, threshold=0.5):
        """
        Make predictions with custom threshold
        """
        probs = self.predict_proba(loader)
        return (probs[:, 1] >= threshold).astype(int)

    def evaluate(self, loader, threshold=0.5):
        """
        Evaluate the model comprehensively
        """
        _, y_test = self.get_latent_features(loader)
        y_test = y_test.cpu().numpy()
        y_pred = self.predict(loader, threshold)

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='binary'
        )

        minority_precision = precision_score(y_test, y_pred, pos_label=1)
        minority_recall = recall_score(y_test, y_pred, pos_label=1)
        minority_f1 = f1_score(y_test, y_pred, pos_label=1)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(report)
        print(cm)
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'minority_precision': minority_precision,
            'minority_recall': minority_recall,
            'minority_f1': minority_f1
        }

    def optimize_threshold(self, val_loader, min_recall=0.3):
        """
        Find optimal threshold based on validation data
        """
        print("\nExtracting validation features for threshold optimization...")
        _, y_val = self.get_latent_features(val_loader, desc="Processing validation data")
        print("\nComputing probabilities...")
        probs = self.predict_proba(val_loader)

        print("\nOptimizing threshold...")
        thresholds = list(np.arange(0.1, 0.9, 0.05))

        best_f1 = 0
        best_threshold = 0.5

        for threshold in tqdm(thresholds, desc="Testing thresholds"):
            y_pred = (probs[:, 1] >= threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, y_pred, average='binary'
            )

            if recall >= min_recall and f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold





# Usage example:
def train_latent_classifier(vae_model, train_loader, val_loader, test_loader, device):
    """Train and evaluate the latent space classifier"""

    # Initialize classifier
    classifier = LatentSpaceClassifier(
        vae_model=vae_model,
        n_neighbors=30,  # Increased for more robust estimation
        contamination=0.01,  # Adjust based on expected anomaly ratio
        device=device
    )

    # Train
    print("Training latent space classifier...")
    classifier.fit(train_loader, val_loader)

    # Find optimal threshold
    best_threshold = classifier.optimize_threshold(
        val_loader,
        min_recall=0.3  # Minimum recall we want to achieve
    )
    print(f"\nOptimal threshold: {best_threshold:.3f}")

    # Evaluate on test set
    test_metrics = classifier.evaluate(test_loader, threshold=best_threshold)
    print("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    return classifier, test_metrics