import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import DataLoader, WeightedRandomSampler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from CNN_hugging_face import create_time_series
from data_process import IdentSubRec

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

TRAIN = False
def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

# Define the feature columns
feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X', 'CURRENT_FIX_IA_Y',
                   'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT']
def apply_input_masking(inputs, mask_probability):
    # Create a mask with the same shape as inputs
    mask = torch.bernoulli(torch.full(inputs.shape, 1 - mask_probability)).to(inputs.device)
    # Apply the mask to the inputs
    masked_inputs = inputs * mask
    return masked_inputs
# Function to split the dataset
def split_train_test(time_series_df):
    df = time_series_df.copy()
    df['group'] = df[['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    split_indices = list(gss.split(X=df[feature_columns], y=df['target'], groups=df['group']))[0]
    train_index, test_index = split_indices
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    return train_df, test_df


# Function to compute class weights
def get_class_weights(Y_train):
    class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
    return {i: class_weights[i] for i in range(len(class_weights))}

# Define the FocalLoss class
class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

# Function to extract encoded features from the autoencoder
def extract_encoded_features(autoencoder, data_loader, device):
    """Extract features from the encoder part of the autoencoder."""
    autoencoder.eval()
    encoded_features = []
    labels = []
    with torch.no_grad():
        for inputs, target in data_loader:
            inputs = inputs.to(device)
            encoded = autoencoder.encoder(inputs)
            encoded = encoded.view(encoded.size(0), -1)  # Flatten the features
            encoded_features.append(encoded.cpu().numpy())
            labels.append(target.cpu().numpy())
    encoded_features = np.concatenate(encoded_features)
    labels = np.concatenate(labels)
    return encoded_features, labels

# Function to perform PCA and t-SNE visualization
def plot_pca_tsne(encoded_features, labels, method='PCA & t-SNE'):
    """Plot PCA and t-SNE visualization for encoded features."""
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(encoded_features)
    print(f'Explained variation per principal component: {pca.explained_variance_ratio_}')

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_result = tsne.fit_transform(encoded_features)

    # Plot PCA
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, palette='viridis')
    plt.title(f'{method} - PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    # Plot t-SNE
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette='viridis')
    plt.title(f'{method} - t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

# Define the Inception Module
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels=32, kernel_sizes=[9, 19, 39]):
        super(InceptionModule, self).__init__()
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, padding=0)
        self.conv_layers = nn.ModuleList()
        for k in kernel_sizes:
            padding = k // 2
            self.conv_layers.append(
                nn.Conv1d(bottleneck_channels, out_channels, kernel_size=k, padding=padding)
            )
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
        )
        self.bn = nn.BatchNorm1d((len(kernel_sizes) + 1) * out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)


    def forward(self, x):
        x_bottleneck = self.bottleneck(x)
        outputs = [conv(x_bottleneck) for conv in self.conv_layers]
        outputs.append(self.maxpool_conv(x))
        x = torch.cat(outputs, dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class InceptionEncoder(nn.Module):
    def __init__(self, in_channels, num_filters, depth):
        super(InceptionEncoder, self).__init__()
        self.blocks = nn.ModuleList()
        self.channels = [in_channels]  # Initialize with input channels
        current_channels = in_channels
        for d in range(depth):
            # InceptionModule outputs num_filters * 4 channels
            self.blocks.append(InceptionModule(current_channels, num_filters))
            current_channels = num_filters * 4
            self.channels.append(current_channels)
            self.blocks.append(nn.MaxPool1d(kernel_size=2))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

    def get_channels(self):
        return self.channels


# Define the Decoder
class InceptionDecoder(nn.Module):
    def __init__(self, num_filters, depth, encoder_channels, input_length):
        super(InceptionDecoder, self).__init__()
        self.blocks = nn.ModuleList()
        decoder_channels = encoder_channels[::-1]
        self.input_length = input_length
        current_length = 1  # Start from the encoded length
        for d in range(depth):
            input_channels = decoder_channels[d]
            output_channels = decoder_channels[d + 1]
            stride = 2
            kernel_size = 2
            # Calculate desired length after this layer
            desired_length = self.input_length // (2 ** (depth - d - 1))
            # Calculate current output length without output_padding
            output_length = (current_length - 1) * stride - 0 + kernel_size
            output_padding = desired_length - output_length
            # Ensure output_padding is non-negative and less than stride
            if output_padding < 0 or output_padding >= stride:
                output_padding = 0
            current_length = desired_length
            self.blocks.append(
                nn.ConvTranspose1d(
                    input_channels,
                    output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    output_padding=output_padding
                )
            )
            self.blocks.append(nn.ReLU())

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# Define the Inception Autoencoder
class InceptionAutoencoder(nn.Module):
    def __init__(self, in_channels, input_length, num_filters=32, depth=3):
        super(InceptionAutoencoder, self).__init__()
        self.encoder = InceptionEncoder(in_channels, num_filters, depth)
        encoder_channels = self.encoder.get_channels()
        self.decoder = InceptionDecoder(num_filters, depth, encoder_channels, input_length)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
# Function to train the autoencoder

def train_autoencoder_with_input_masking(model, train_loader, val_loader, criterion, optimizer, epochs, device, scheduler, mask_probability=0.1, early_stopping_patience=100):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()

            z = apply_input_masking(inputs, mask_probability)

            outputs = model(z)

            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation and logging
        avg_train_loss = running_loss / len(train_loader)
        val_loss = validate_autoencoder(model, val_loader, criterion, epoch, running_loss, train_loader, device, mask_probability=mask_probability)
        scheduler.step(val_loss)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_autoencoder.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f'Early stopping after {epoch + 1} epochs')
                # Load the best model weights
                model.load_state_dict(torch.load('best_autoencoder.pth'))
                break

    # Plot the training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Autoencoder Training and Validation Loss')
    plt.legend()
    plt.show()


def train_autoencoder(model, train_loader, val_loader, criterion, optimizer, epochs, device,scheduler):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)  # Compare output with the original input
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 10 == 0:
            val_loss=validate_autoencoder(model, val_loader, criterion, epoch, running_loss, train_loader, device)
            scheduler.step(val_loss)

# Function to validate the autoencoder
def validate_autoencoder(model, val_loader, criterion, epoch, running_loss, train_loader, device, mask_probability=-1.0):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            if mask_probability>0:
                inputs = apply_input_masking(inputs, mask_probability)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            val_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    return avg_val_loss
# Define the classification model using extracted features
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Function to train the classifier
def train_classifier(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Optionally, evaluate on validation set
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')
            evaluate_classifier(model, val_loader, device)

# Function to evaluate the classifier
def evaluate_classifier(model, val_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())
    # Calculate metrics
    print(classification_report(y_true, y_pred))
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()

# Main function integrating everything
def main_with_autoencoder(df, window_size=5, method='', resample=True, epochs=200, batch_size=32, ae_epochs=100):
    interval = '30ms'
    # Step 1: Split the dataset
    train_df, test_df = split_train_test(df)
    scaler = StandardScaler()
    scaler.fit(train_df[feature_columns])
    # Step 2: Create time series data for training and testing
    X_train, Y_train, window_weight_train = create_time_series(train_df, interval, window_size=window_size, resample=resample)
    X_test, Y_test, window_weights_test = create_time_series(test_df, interval, window_size=window_size, resample=resample)

    num_samples, seq_len, num_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, num_features)
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(num_samples, seq_len, num_features)
    num_samples_test, seq_len_test, num_features_test = X_test.shape
    X_test_reshaped = X_test.reshape(-1, num_features_test)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(num_samples_test, seq_len_test, num_features_test)

    # Step 4: Prepare data loaders for the autoencoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).permute(0, 2, 1)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    train_dataset_ae = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset_ae = torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor)
    train_loader_ae = torch.utils.data.DataLoader(train_dataset_ae, batch_size=batch_size, shuffle=True)
    # Compute sample weights

    train_labels_numpy = Y_train  # Ensure Y_train is a NumPy array
    classes, class_indices = np.unique(train_labels_numpy, return_inverse=True)
    class_sample_counts = np.array([np.sum(class_indices == i) for i in range(len(classes))])
    weights = 1. / class_sample_counts

    samples_weight = weights[class_indices]
    samples_weight = torch.from_numpy(samples_weight).float()

    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader_ae = torch.utils.data.DataLoader(train_dataset_ae, batch_size=batch_size, sampler=sampler)

    test_loader_ae = torch.utils.data.DataLoader(test_dataset_ae, batch_size=batch_size, shuffle=False)

    # Step 5: Initialize and train the autoencoder
    in_channels = len(feature_columns)
    input_length = X_train_tensor.shape[2]  # Get the sequence length from the input tensor
    autoencoder = InceptionAutoencoder(in_channels=in_channels, input_length=input_length, num_filters=32, depth=4).to(
        device)
    autoencoder.apply(initialize_weights)

    criterion_ae = nn.MSELoss()
    optimizer_ae = optim.Adam(autoencoder.parameters(), lr=0.01, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ae, mode='min', factor=0.5,patience=20)

    # train_autoencoder(autoencoder, train_loader_ae, test_loader_ae, criterion_ae, optimizer_ae, epochs=ae_epochs,
    #                   device=device,scheduler=scheduler)
    if TRAIN:

        train_autoencoder_with_input_masking(autoencoder, train_loader_ae, test_loader_ae, criterion_ae, optimizer_ae,
                                             epochs=ae_epochs, device=device,scheduler=scheduler, mask_probability=0.4)
    torch.save(autoencoder.state_dict(), 'best_autoencoder.pth')
    autoencoder.load_state_dict(torch.load('best_autoencoder.pth'))

    encoded_features_train, train_labels = extract_encoded_features(autoencoder, train_loader_ae, device)
    encoded_features_test, test_labels = extract_encoded_features(autoencoder, test_loader_ae, device)

    print("Visualizing encoded features using PCA and t-SNE...")
    plot_pca_tsne(encoded_features_train, train_labels, method='Autoencoder - Train Data')
    plot_pca_tsne(encoded_features_test, test_labels, method='Autoencoder - Test Data')
    #
    # # Step 8: Prepare data loaders for the classifier
    # train_dataset_cls = torch.utils.data.TensorDataset(torch.tensor(encoded_features_train, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long))
    # test_dataset_cls = torch.utils.data.TensorDataset(torch.tensor(encoded_features_test, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long))
    # train_loader_cls = torch.utils.data.DataLoader(train_dataset_cls, batch_size=batch_size, shuffle=True)
    # test_loader_cls = torch.utils.data.DataLoader(test_dataset_cls, batch_size=batch_size, shuffle=False)
    #
    # # Step 9: Initialize and train the classifier
    # input_dim = encoded_features_train.shape[1]
    # classifier = ClassificationHead(input_dim=input_dim, num_classes=2).to(device)
    # class_weights_dict = get_class_weights(Y_train)
    # class_weights = torch.tensor([class_weights_dict[0], class_weights_dict[1]], dtype=torch.float32).to(device)
    # criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    # optimizer_cls = optim.Adam(classifier.parameters(), lr=0.001)
    # train_classifier(classifier, train_loader_cls, test_loader_cls, criterion_cls, optimizer_cls, epochs=epochs, device=device)
    #
    # # Step 10: Evaluate the model
    # print("Evaluating the classifier on the test set...")
    # evaluate_classifier(classifier, test_loader_cls, device)

if __name__ == '__main__':
    # Load your dataset
    categorized_rad_s1_s18_file_path = 'data/Categorized_Fixation_Data_1_18.csv'
    categorized_rad_init_kwargs = dict(
        data_file_path=categorized_rad_s1_s18_file_path,
        test_data_file_path=None,
        augment=False,
        join_train_and_test_data=False,
        normalize=True,
        remove_surrounding_to_hits=0,
        update_surrounding_to_hits=0,
        approach_num=6,
    )

    ident_sub_rec = IdentSubRec(**categorized_rad_init_kwargs)
    df = ident_sub_rec.df
    window = 100

    main_with_autoencoder(df, window_size=window, method='autoencoder_classification')
