import os.path

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster._hdbscan import hdbscan
from sklearn.decomposition import PCA
import matplotlib
from sklearn.manifold import TSNE
from sklearn.svm import SVC


import matplotlib.pyplot as plt
from matplotlib import colormaps
from tabpfn import TabPFNClassifier
from torch.utils.data import WeightedRandomSampler, DataLoader, Dataset

cmap = colormaps['tab10']
from representation_method.utils.data_loader import DataConfig, load_eye_tracking_data
from representation_method.utils.data_utils import create_dynamic_time_series, split_train_test_for_time_series
from sklearn.cluster import KMeans

from ruptures.detection import Binseg
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F


from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)
import seaborn as sns


# Step 4: Plot Decision Boundary and Clusters
def plot_svm_high_dim(latent_2d, true_labels, predictions, decision_boundary, split, seed=0):
    """
    Visualize SVM decision boundary (trained in high-dimensional space) in 2D.

    Parameters:
        latent_2d (ndarray): PCA-reduced 2D latent space.
        true_labels (ndarray): Ground truth target labels.
        predictions (ndarray): SVM predictions.
        decision_boundary (ndarray): 2D SVM decision boundary grid.
        split (str): 'train' or 'test' for plot title.
    """
    plt.figure(figsize=(10, 8))

    # Plot decision boundary
    plt.contourf(xx, yy, decision_boundary, alpha=0.3, cmap='coolwarm')

    # Plot true labels (always shown under predictions)
    plt.scatter(
        latent_2d[true_labels == 0, 0],
        latent_2d[true_labels == 0, 1],
        color='blue',
        label="True Label 0",
        alpha=0.8,
        marker='X',
        s=80
    )
    plt.scatter(
        latent_2d[true_labels == 1, 0],
        latent_2d[true_labels == 1, 1],
        color='red',
        label="True Label 1",
        alpha=0.8,
        marker='X',
        s=80
    )

    # Highlight SVM predictions
    plt.scatter(
        latent_2d[predictions == 1, 0],
        latent_2d[predictions == 1, 1],
        facecolor='none',
        marker='o',
        s=120,
        linewidth=1.5,
        alpha=0.2,

        label="SVM Positive Predictions"
    )
    plt.scatter(
        latent_2d[predictions == 0, 0],
        latent_2d[predictions == 0, 1],
        facecolor='none',
        marker='o',
        s=120,
        linewidth=1.5,
        alpha=0.2,

        label="SVM Negative Predictions"
    )

    plt.title(f"SVM Decision Boundary (High Dim to 2D) ({split})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

class AggDataset(Dataset):
    def __init__(self, df, feature_columns, label_column='target'):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the features and labels.
            feature_columns (list): List of column names to use as input features.
            label_column (str): Name of the column containing labels.
        """
        self.features = torch.tensor(df[feature_columns].values, dtype=torch.float)
        self.labels = torch.tensor(df[label_column].values, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def get_triplet(self, batch_data, batch_labels):

        anchor = batch_data
        positive = torch.roll(batch_data, shifts=1, dims=0)
        negative = batch_data.flip(0)
        return anchor, positive, negative




class TripletAutoencoder(nn.Module):
    def __init__(self, input_dim=40, latent_dim=8, dropout_rate=0.5, activation='ReLU', normalize_latent=True):
        """
        Enhanced Triplet Autoencoder with residual connections, flexible activation, and latent space normalization.
        """
        super().__init__()
        self.normalize_latent = normalize_latent

        # Select activation function
        activation_fn = {
            'ReLU': nn.ReLU,
            'LeakyReLU': nn.LeakyReLU,
            'GELU': nn.GELU
        }[activation]

        # Encoder with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, latent_dim)
        )

        # Decoder with residual connections
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, input_dim)
        )

        # Classifier for target vs. non-target (assumes 2 classes)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        # Optionally add noise during training for regularization
        if self.training:
            noise_scale = 0.1
            x = x + torch.randn_like(x) * noise_scale

        z = self.encoder(x)  # Latent embedding
        if len(z.shape) == 1:  # if z is 1D
            z = z.unsqueeze(0)
        if self.normalize_latent:
            z = F.normalize(z, p=2, dim=1)  # normalize embeddings to unit length

        x_recon = self.decoder(z)  # Reconstructed input
        logits = self.classifier(z)  # Classification logits

        return x_recon, z, logits

    def encode(self, x):
        z = self.encoder(x)
        return F.normalize(z, p=2, dim=1) if self.normalize_latent else z

    def decode(self, z):
        return self.decoder(z)

# ---------------------------
# Define the training function for the Triplet AE
# ---------------------------
def train_triplet_ae(
    train_dataset,
    val_dataset,
    input_dim,
    latent_dim=8,
    margin=1.0,
    lambda_recon=1.0,
    lambda_triplet=1.0,
    lambda_clf=2.0,
    batch_size=64,
    epochs=10,
    lr=1e-3,
    device="cpu"
):
    """
    Train a triplet autoencoder that combines reconstruction, triplet, and classification losses.
    """
    # Create DataLoaders with weighted sampling to mitigate class imbalance.
    labels = train_dataset.labels.numpy()  # Assumes train_dataset.labels is a tensor with class labels
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss functions
    model = TripletAutoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    best_model = model
    best_val_loss = float('inf')

    # For plotting training curves
    history = {
        "train_recon": [],
        "train_triplet": [],
        "train_clf": [],
        "val_recon": [],
        "val_triplet": [],
        "val_clf": [],
        "val_total": []
    }

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_recon_loss = 0.0
        train_triplet_loss = 0.0
        train_clf_loss = 0.0
        num_train_batches = 0

        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)   # Input features
            batch_labels = batch_labels.to(device) # Target labels (for classifier)

            # Forward pass on full batch for reconstruction and classification
            x_recon, z_batch, logits_batch = model(batch_data)
            clf_loss = F.cross_entropy(logits_batch, batch_labels)

            # Get triplets from the dataset (anchor, positive, negative)
            anchor, positive, negative = train_dataset.get_triplet(batch_data, batch_labels)
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # Forward pass on triplets to get embeddings
            _, z_anchor, _ = model(anchor)
            _, z_positive, _ = model(positive)
            _, z_negative, _ = model(negative)

            trip_loss = triplet_loss_fn(z_anchor, z_positive, z_negative)
            recon_loss = mse_loss_fn(x_recon, batch_data)

            loss = lambda_recon * recon_loss + lambda_triplet * trip_loss + lambda_clf * clf_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_recon_loss += recon_loss.item()
            train_triplet_loss += trip_loss.item()
            train_clf_loss += clf_loss.item()
            num_train_batches += 1

        avg_train_recon = train_recon_loss / num_train_batches
        avg_train_triplet = train_triplet_loss / num_train_batches
        avg_train_clf = train_clf_loss / num_train_batches

        # --- Validation ---
        model.eval()
        val_recon_loss = 0.0
        val_triplet_loss = 0.0
        val_clf_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                x_recon, z_batch, logits_batch = model(batch_data)
                clf_loss = F.cross_entropy(logits_batch, batch_labels)
                recon_loss = mse_loss_fn(x_recon, batch_data)
                val_recon_loss += recon_loss.item()

                # Get triplets for validation
                anchor, positive, negative = val_dataset.get_triplet(batch_data, batch_labels)
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                _, z_anchor, _ = model(anchor)
                _, z_positive, _ = model(positive)
                _, z_negative, _ = model(negative)
                trip_loss = triplet_loss_fn(z_anchor, z_positive, z_negative)
                val_triplet_loss += trip_loss.item()

                val_clf_loss += clf_loss.item()
                num_val_batches += 1

        avg_val_recon = val_recon_loss / num_val_batches
        avg_val_triplet = val_triplet_loss / num_val_batches
        avg_val_clf = val_clf_loss / num_val_batches
        avg_val_total = lambda_recon * avg_val_recon + lambda_triplet * avg_val_triplet + lambda_clf * avg_val_clf

        # Save best model
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            best_model = model
            torch.save(model.state_dict(), "best_triplet_ae_model.pt")

        # Store history for plotting later
        history["train_recon"].append(avg_train_recon)
        history["train_triplet"].append(avg_train_triplet)
        history["train_clf"].append(avg_train_clf)
        history["val_recon"].append(avg_val_recon)
        history["val_triplet"].append(avg_val_triplet)
        history["val_clf"].append(avg_val_clf)
        history["val_total"].append(avg_val_total)

        print(f"Epoch [{epoch+1}/{epochs}] Train: Recon {avg_train_recon:.3f}, Triplet {avg_train_triplet:.3f}, Clf {avg_train_clf:.3f} | "
              f"Val: Recon {avg_val_recon:.3f}, Triplet {avg_val_triplet:.3f}, Clf {avg_val_clf:.3f}, Total {avg_val_total:.3f}")

    print("Training complete. Best val loss:", best_val_loss)
    return best_model, history

def explore_agg_data(aggregated_features,feature_to_test):
    # Create subplots: one row per feature
    num_features = len(feature_to_test)
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(8, num_features * 3))
    if num_features == 1:
        axes = [axes]
    for ax, feature in zip(axes, feature_to_test):
        sns.histplot(df_target_0[feature], color='blue', kde=True, stat="density",
                     label='target 0', ax=ax, bins=20, alpha=0.6)
        sns.histplot(df_target_1[feature], color='red', kde=True, stat="density",
                     label='target 1', ax=ax, bins=20, alpha=0.6)

        ax.set_title(f'Distribution of {feature}')
        ax.legend()
    plt.tight_layout()
    plt.show()
    X = aggregated_features[feature_to_test].values
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(X)
    aggregated_features['tsne-2d-one'] = tsne_result[:, 0]
    aggregated_features['tsne-2d-two'] = tsne_result[:, 1]
    colors = {0: 'blue', 1: 'red'}
    plt.figure(figsize=(8, 8))
    for target_value in aggregated_features['target'].unique():
        indices = aggregated_features['target'] == target_value
        plt.scatter(aggregated_features.loc[indices, 'tsne-2d-one'],
                    aggregated_features.loc[indices, 'tsne-2d-two'],
                    c=colors[target_value],
                    label=f'target {target_value}',
                    alpha=0.6)
    plt.title("t-SNE Cluster Plot")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.show()


def lgm_cls(X_train,X_test,y_train,y_test,X_val,y_val,class_weight):
    # Initialize the LGBMClassifier with the specified hyperparameters.
    model = LGBMClassifier(
        n_estimators=100,
        class_weight=class_weight,
        random_state=seed,
        num_leaves=31,
        learning_rate=0.1,
        min_child_samples=20,
        force_row_wise=True,
        verbose=-1,
    )
    # Train the model with early stopping on the validation set.
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc'
    )
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    # Calculate evaluation metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    print("Test AUC:", auc)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    # ---- Plot Confusion Matrix ----
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax_cm)
    ax_cm.set_title("Confusion Matrix")
    plt.show()
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    # ---- Plot Feature Importances ----
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), importances[indices], color='green', align='center')
    plt.yticks(range(len(indices)), [feature_columns[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.show()


if __name__ == '__main__':
    seed = 0
    feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'relative_x', 'relative_y',
                       'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT', 'gaze_velocity',"target"]
    # ------------------------------------------------------------
    # Example usage within your pipeline:
    config = DataConfig(
        data_path='data/Categorized_Fixation_Data_1_18.csv',
        approach_num=8,
        normalize=True,
        per_slice_target=False,
        participant_id=1
    )

    # Load legacy data
    df = load_eye_tracking_data(
        data_path=config.data_path,
        approach_num=config.approach_num,
        participant_id=config.participant_id,
        data_format="legacy"
    )


    df['fix_group'] = (df['CURRENT_FIX_COMPONENT_INDEX'] == 1).cumsum()

    agg_methods = {
        'Pupil_Size': 'mean',
        'CURRENT_FIX_DURATION': 'mean',
        'relative_x': 'mean',
        'relative_y': 'mean',
        'CURRENT_FIX_INDEX': 'mean',
        'CURRENT_FIX_COMPONENT_COUNT': 'mean',
        'gaze_velocity': 'mean',
        'target': 'max'
    }

    aggregated_features = df.groupby('fix_group')[feature_columns].agg(agg_methods).reset_index(drop=True)
    df_target_1 = aggregated_features[aggregated_features['target'] == 1]
    df_target_0 = aggregated_features[aggregated_features['target'] == 0]
    feature_to_test = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'relative_x', 'relative_y',
                        'CURRENT_FIX_COMPONENT_COUNT', 'gaze_velocity']
    explore_agg_data(aggregated_features, feature_to_test)
    train_df, test_df = split_train_test_for_time_series(df, test_size=0.2, random_state=seed)
    train_df, val_df = split_train_test_for_time_series(train_df, test_size=0.2, random_state=seed)


    feature_columns = [
        'Pupil_Size', 'CURRENT_FIX_DURATION', 'relative_x', 'relative_y',
        'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT', 'gaze_velocity'
    ]

    # Assume you have a seed and class_weight defined for handling imbalance.
    seed = 42
    # For example, you can compute class weights or define them manually.
    # Here, assume class_weight is a dict like: {0: 1, 1: 10}
    class_weight = {0: 1, 1: 10}

    # Split the data using your time series split function
    train_df, test_df = split_train_test_for_time_series(df, test_size=0.2, random_state=seed)
    train_df, val_df = split_train_test_for_time_series(train_df, test_size=0.2, random_state=seed)
    train_dataset = AggDataset(train_df, feature_columns, label_column='target')
    val_dataset = AggDataset(val_df, feature_columns, label_column='target')
    test_dataset = AggDataset(test_df, feature_columns, label_column='target')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model
    model, history = train_triplet_ae(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_dim=len(feature_columns),
        latent_dim=8,
        margin=1.0,
        lambda_recon=1.0,
        lambda_triplet=1.0,
        lambda_clf=2.0,
        batch_size=64,
        epochs=100,
        lr=1e-3,
        device="cuda"
    )


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    def extract_features(data_loader):
        features_list = []
        labels_list = []
        for batch_features, labels in data_loader:  # Assuming each item is a (features, label) tuple
            # If features are already tensors, just convert to numpy later.
            features_list.append(batch_features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

        # Concatenate along the first dimension (samples)
        return np.concatenate(features_list, axis=0),np.concatenate(labels_list, axis=0)

    train_features,train_labels = extract_features(train_loader)
    test_features,test_labels = extract_features(test_loader)

    # Now, get the latent representations using your model's encoder.
    with torch.no_grad():
        latent_train = model.encoder(torch.from_numpy(train_features).float().to(device)).cpu().numpy()
        latent_test = model.encoder(torch.from_numpy(test_features).float().to(device)).cpu().numpy()

    svm_far = SVC(
        kernel='rbf',
        C=1,  # Slightly easier regularization
        gamma=0.5,  # A lower gamma can produce a smoother decision boundary
        probability=True,
        random_state=42,
        class_weight='balanced'
    )
    svm_far.fit(latent_train, train_labels)

    # Get SVM predictions on the training set (for visualization)
    train_predictions = svm_far.predict(latent_train)
    test_predictions = svm_far.predict(latent_test)

    # ---------------------------
    # 3. Reduce Latent Representations to 2D for Plotting via PCA
    # ---------------------------
    pca = PCA(n_components=2)
    latent_train_2d = pca.fit_transform(latent_train)
    latent_test_2d = pca.fit_transform(latent_test)

    # Create a grid over the 2D space
    x_min, x_max = latent_train_2d[:, 0].min() - 1, latent_train_2d[:, 0].max() + 1
    y_min, y_max = latent_train_2d[:, 1].min() - 1, latent_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_2d = np.c_[xx.ravel(), yy.ravel()]

    # Map the 2D grid back to the original latent space using PCA inverse transform.
    # This gives us points in the original latent space on which we can evaluate the SVM.
    grid_high_dim = pca.inverse_transform(grid_2d)
    Z = svm_far.decision_function(grid_high_dim)
    Z = Z.reshape(xx.shape)

    plot_svm_high_dim(
        latent_train_2d,
        true_labels=train_labels,
        predictions=train_predictions,
        decision_boundary=Z,
        split="train",
        seed=seed
    )

    plot_svm_high_dim(
        latent_test_2d,
        true_labels=test_labels,
        predictions=test_predictions,
        decision_boundary=Z,
        split="test",
        seed=seed
    )







    # ---------------------------
    # Plotting Loss Curves
    # ---------------------------
    epochs_range = range(1, len(history["train_recon"]) + 1)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, history["train_recon"], label="Train Recon Loss", marker='o')
    plt.plot(epochs_range, history["val_recon"], label="Val Recon Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Reconstruction Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, history["train_triplet"], label="Train Triplet Loss", marker='o')
    plt.plot(epochs_range, history["val_triplet"], label="Val Triplet Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Triplet Loss")
    plt.title("Triplet Loss")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, history["train_clf"], label="Train Clf Loss", marker='o')
    plt.plot(epochs_range, history["val_clf"], label="Val Clf Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Classification Loss")
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, history["val_total"], label="Val Total Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Combined Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # ---------------------------
    # Evaluation on Test Set (Classifier Performance)
    # ---------------------------
    # Create a DataLoader for the test set
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    all_labels = []
    all_preds = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            # Get classification logits from the model
            _, _, logits = model(batch_data)
            probs = F.softmax(logits, dim=1)[:, 1]  # probability for class 1
            preds = torch.argmax(logits, dim=1)
            all_labels.append(batch_labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    # Classification Report
    print("Test Classification Report:\n", classification_report(all_labels, all_preds))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax_cm)
    ax_cm.set_title("Test Confusion Matrix")
    plt.show()

    # ROC Curve
    auc = roc_auc_score(all_labels, all_probs)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Test ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    #

    # # Define your feature columns and target column
    # feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'relative_x', 'relative_y',
    #                    'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT', 'gaze_velocity']
    # target_column = 'target'
    #
    # # Extract features and targets
    # X_train = train_df[feature_columns].values
    # y_train = train_df[target_column].values
    #
    # X_test = test_df[feature_columns].values
    # y_test = test_df[target_column].values
    #
    # # Initialize and fit TabPFNClassifier
    # clf = TabPFNClassifier(random_state=42,ignore_pretraining_limits=True)
    # clf.fit(X_train, y_train)
    #
    # # Predict probabilities on the test set
    # y_pred_proba = clf.predict_proba(X_test)
    #
    # # (Optional) You can also get hard predictions if needed:
    # y_pred = clf.predict(X_test)
    #
    # # Print out the predicted probabilities and some evaluation metrics
    # print("Predicted probabilities:\n", y_pred_proba)
    # print("\nClassification Report:\n", classification_report(y_test, y_pred))
    # print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))