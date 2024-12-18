import os
import random

import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GroupShuffleSplit, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from torch.utils.data import DataLoader, WeightedRandomSampler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import cnn_time_series
from TimeGanOriginal.timegan import timegan
from cnn_time_series import create_time_series, validate_model, plot_confusion_matrix
from data_process import IdentSubRec

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt #3.4.0
import torch
from torchsampler import ImbalancedDatasetSampler


TRAIN = False


device = "cuda"
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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


# Function to compute class weights
def get_class_weights(labels):
    """Compute class weights for imbalanced datasets."""
    # Check if labels is a PyTorch tensor
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()  # Move to CPU and convert to NumPy array
    else:
        labels = np.array(labels)  # Ensure labels is a NumPy array

    classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=labels)
    class_weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
    return class_weights_dict
# Define the FocalLoss class

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, weights):
        log_probs = torch.nn.functional.log_softmax(inputs, dim=1)
        weighted_loss = -weights * log_probs[range(len(targets)), targets]
        return weighted_loss.mean()




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
def plot_pca_tsne(encoded_features, labels, method='PCA & t-SNE', save_path=''):
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
    # plt.show()
    plt.savefig(os.path.join(save_path, f'{method}_PCA.png'))
    plt.close()

    # Plot t-SNE
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels, palette='viridis')
    plt.title(f'{method} - t-SNE')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.savefig(os.path.join(save_path, f'{method}_tSNE.png'))
    plt.close()


class CNNRecurrentEncoder(nn.Module):
    def __init__(self, in_channels, num_filters, depth, hidden_size, num_layers=1, rnn_type='GRU'):
        super(CNNRecurrentEncoder, self).__init__()
        self.conv_encoder = InceptionEncoder(in_channels, num_filters, depth)
        self.encoder_channels = self.conv_encoder.get_channels()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=self.encoder_channels[-1], hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.encoder_channels[-1], hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'GRU' or 'LSTM'")

        self.encoded_length = None

    def forward(self, x):
        x = self.conv_encoder(x)
        self.encoded_length = x.size(2)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)
        outputs, hidden = self.rnn(x)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return outputs, hidden

    def get_channels(self):
        return self.encoder_channels

    def get_encoded_length(self):
        return self.encoded_length



class CNNRecurrentDecoder(nn.Module):
    def __init__(self, num_filters, depth, encoder_channels, input_length, hidden_size, num_layers=1, rnn_type='GRU'):
        super(CNNRecurrentDecoder, self).__init__()
        self.hidden_size = hidden_size

        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=hidden_size, hidden_size=encoder_channels[-1], num_layers=num_layers, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=encoder_channels[-1], num_layers=num_layers, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'GRU' or 'LSTM'")

        self.conv_decoder = InceptionDecoder(num_filters=num_filters, depth=depth, encoder_channels=encoder_channels, input_length=input_length)

    def forward(self, x):

        outputs, hidden = self.rnn(x)
        x = outputs.permute(0, 2, 1)
        x = self.conv_decoder(x)
        return x


class CNNRecurrentAutoencoder(nn.Module):
    def __init__(self, in_channels, num_filters, depth, hidden_size, num_layers=1, rnn_type='GRU', input_length=None):
        super(CNNRecurrentAutoencoder, self).__init__()
        self.encoder = CNNRecurrentEncoder(
            in_channels=in_channels,
            num_filters=num_filters,
            depth=depth,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type
        )
        encoder_channels = self.encoder.get_channels()
        if input_length is None:
            input_length = self.encoder.get_encoded_length()
        self.decoder = CNNRecurrentDecoder(
            num_filters=num_filters,
            depth=depth,
            encoder_channels=encoder_channels,
            input_length=input_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type
        )

    def forward(self, x):
        encoded_outputs, hidden = self.encoder(x)
        reconstructed_outputs = self.decoder(encoded_outputs)
        return reconstructed_outputs


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

class IndependentInceptionModule(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[9, 19, 39]):
        super(IndependentInceptionModule, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.kernel_sizes = kernel_sizes
        for k in kernel_sizes:
            padding = k // 2
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=in_channels,  # Keep output channels equal to input channels
                    kernel_size=k,
                    padding=padding,
                    groups=in_channels  # Depthwise convolution
                )
            )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm1d(in_channels * (len(kernel_sizes) + 1))
        self.channel_reduction = nn.Conv1d(
            in_channels=in_channels * (len(kernel_sizes) + 1),
            out_channels=in_channels,
            kernel_size=1,
            groups=in_channels  # Keep features independent
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        outputs = [conv(x) for conv in self.conv_layers]
        outputs.append(self.maxpool(x))
        x = torch.cat(outputs, dim=1)
        x = self.bn(x)
        x = self.channel_reduction(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x



class IndependentInceptionEncoder(nn.Module):
    def __init__(self, in_channels, depth, kernel_sizes=[9, 19, 39]):
        super(IndependentInceptionEncoder, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(IndependentInceptionModule(in_channels, kernel_sizes=kernel_sizes))
            self.blocks.append(nn.MaxPool1d(kernel_size=2))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x



class IndependentInceptionDecoder(nn.Module):
    def __init__(self, in_channels, depth, kernel_sizes=[9, 19, 39]):
        super(IndependentInceptionDecoder, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ConvTranspose1d(
                in_channels,
                in_channels,
                kernel_size=2,
                stride=2,
                groups=in_channels  # Keep features independent
            ))
            self.blocks.append(IndependentInceptionModule(in_channels, kernel_sizes=kernel_sizes))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class IndependentInceptionAutoencoder(nn.Module):
    def __init__(self, in_channels, input_length, depth=3, kernel_sizes=[9, 19, 39]):
        super(IndependentInceptionAutoencoder, self).__init__()
        self.encoder = IndependentInceptionEncoder(in_channels, depth, kernel_sizes=kernel_sizes)
        self.decoder = IndependentInceptionDecoder(in_channels, depth, kernel_sizes=kernel_sizes)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.projection = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.projection = None

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.projection is not None:
            residual = self.projection(residual)
        out += residual  # Add skip connection
        out = self.relu(out)
        return out


class ComplexCNNClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ComplexCNNClassifier, self).__init__()
        # Ensure in_channels=1
        self.initial_conv = nn.Conv1d(128, 64, kernel_size=7, padding=3)
        self.initial_bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)

        # Define residual blocks with correct channel transitions
        self.layer1 = self._make_layer(ResidualBlock, in_channels=64, out_channels=64, num_blocks=2, dilation=1)
        self.layer2 = self._make_layer(ResidualBlock, in_channels=64, out_channels=128, num_blocks=2, dilation=2)
        self.layer3 = self._make_layer(ResidualBlock, in_channels=128, out_channels=256, num_blocks=2, dilation=4)
        self.layer4 = self._make_layer(ResidualBlock, in_channels=256, out_channels=512, num_blocks=2, dilation=8)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, in_channels, out_channels, num_blocks, dilation):
        layers = []
        layers.append(block(in_channels, out_channels, dilation=dilation))
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.initial_bn(self.initial_conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x





def train_autoencoder_with_input_masking(model, train_loader, val_loader, criterion, optimizer, epochs, device, scheduler, mask_probability=0.1,  save_path='',early_stopping_patience=100):
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

            z =inputs
            # z = apply_input_masking(inputs, mask_probability)
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

    plt.savefig(os.path.join(save_path, 'training_loss.png'))
    plt.close()



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


def train_cnn(model, train_loader, val_loader, criterion, optimizer, epochs, device, early_stopping_patience=10,save_path =''):
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)  # No need to permute dimensions
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validate the model
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_labels).item()
        val_loss /= len(val_loader)

        # Print training and validation loss
        print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")




def evaluate_model(model, test_loader, device,save_path='',threshold = 0.5):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)[:, 0]
            probabilities = torch.sigmoid(outputs).squeeze()
            threshold = threshold
            preds = (probabilities >= threshold).long()
            # _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    print("Classification Report:")
    classification_rep = classification_report(all_labels, all_preds)
    print(classification_rep)
    # plot_confusion_matrix(all_labels, all_preds)
    num_preds = len(all_preds)
    num_labels = len(all_labels)
    cm = confusion_matrix(all_labels, all_preds)

    print("Confusion Matrix:")
    print("   Predicted 0  Predicted 1")
    print(f"Actual 0   {cm[0, 0]:<10} {cm[0, 1]:<10}")
    print(f"Actual 1   {cm[1, 0]:<10} {cm[1, 1]:<10}")

    if save_path != '':
        with open(save_path, 'w') as f:
            f.write("Classification Report:\n")
            f.write(classification_rep)
            f.write("\n")
            f.write(f"Number of Predictions: {num_preds}\n")
            f.write(f"Number of Labels: {num_labels}\n")
            f.write("Confusion Matrix:\n")
            f.write("   Predicted 0  Predicted 1\n")
            f.write(f"Actual 0   {cm[0, 0]:<10} {cm[0, 1]:<10}\n")
            f.write(f"Actual 1   {cm[1, 0]:<10} {cm[1, 1]:<10}\n")


def train_combined_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, scheduler,
                         save_path, early_stopping_patience, window_weight_train_tensor, window_weight_val_tensor):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            weights = window_weight_train_tensor[i].to(device)  # Use window weights for training

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels, weights)  # Pass the weights to the custom criterion

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        val_running_loss = 0.0
        correct = 0

        with torch.no_grad():
            for j, (val_inputs, val_labels) in enumerate(val_loader):
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                val_weights = window_weight_val_tensor[j].to(device)  # Use window weights for validation

                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels, val_weights)  # Pass the weights

                val_running_loss += val_loss.item() * val_inputs.size(0)

                _, preds = torch.max(val_outputs, 1)
                correct += torch.sum(preds == val_labels.data)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = correct.double() / len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {epoch_loss:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, '
              f'Val Acc: {val_accuracy:.4f}')

        scheduler.step(val_epoch_loss)

        # Early Stopping
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(save_path, 'best_combined_model.pth'))
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break


def extract_encoded_features_from_combined_model(model, data_loader, device):
    model.eval()
    encoded_features = []
    labels = []
    with torch.no_grad():
        for inputs, lbls in data_loader:
            inputs = inputs.to(device)
            lbls = lbls.numpy()
            # Pass inputs through the encoder part of the model
            encoded = model.encoder(inputs)
            # Flatten the encoded features if necessary
            encoded = encoded.view(encoded.size(0), -1)
            encoded_features.append(encoded.cpu().numpy())
            labels.append(lbls)
    encoded_features = np.concatenate(encoded_features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return encoded_features, labels

    # 7. Combine Encoder and Classifier
class CombinedModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super(CombinedModel, self).__init__()
        self.encoder = encoder.encoder  # Use the encoder part of the autoencoder
        self.classifier = ComplexCNNClassifier(input_dim=128,num_classes=num_classes)

    def forward(self, x):
        outputs, hidden = self.encoder(x)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        x = outputs.permute(0, 2, 1)
        logits = self.classifier(x)
        return logits


def validate_synthetic_data(original_data, synthetic_data, save_dir):
    """
    Comprehensive validation of synthetic data quality through various metrics and visualizations.

    Args:
        original_data: Original minority class data (numpy array of shape [n_samples, seq_len, n_features])
        synthetic_data: Generated synthetic data (same shape as original_data)
        save_dir: Directory to save validation results_old and plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Open a file to write statistical results_old
    with open(os.path.join(save_dir, 'validation_metrics.txt'), 'w') as f:
        f.write("Synthetic Data Validation Metrics\n")
        f.write("================================\n\n")

        # Basic shape information
        f.write(f"Data Shapes:\n")
        f.write(f"Original data: {original_data.shape}\n")
        f.write(f"Synthetic data: {synthetic_data.shape}\n\n")

        # Statistical moments for each feature
        f.write("Statistical Moments per Feature:\n")
        for feature_idx in range(original_data.shape[2]):
            orig_feat = original_data[:, :, feature_idx].flatten()
            syn_feat = synthetic_data[:, :, feature_idx].flatten()

            f.write(f"\nFeature {feature_idx}:\n")
            f.write(f"Mean - Original: {np.mean(orig_feat):.4f}, Synthetic: {np.mean(syn_feat):.4f}\n")
            f.write(f"Std  - Original: {np.std(orig_feat):.4f}, Synthetic: {np.std(syn_feat):.4f}\n")
            f.write(
                f"Skew - Original: {scipy.stats.skew(orig_feat):.4f}, Synthetic: {scipy.stats.skew(syn_feat):.4f}\n")
            f.write(
                f"Kurt - Original: {scipy.stats.kurtosis(orig_feat):.4f}, Synthetic: {scipy.stats.kurtosis(syn_feat):.4f}\n")

    # Create visualizations
    _plot_feature_distributions(original_data, synthetic_data, save_dir)
    _plot_temporal_patterns(original_data, synthetic_data, save_dir)
    _plot_correlation_matrices(original_data, synthetic_data, save_dir)
    _plot_pca_analysis(original_data, synthetic_data, save_dir)


def _plot_feature_distributions(original_data, synthetic_data, save_dir):
    """Plot distribution comparisons for each feature."""
    n_features = original_data.shape[2]
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    for i in range(n_features):
        plt.subplot(n_rows, n_cols, i + 1)

        # Plot histograms
        plt.hist(original_data[:, :, i].flatten(), bins=50, alpha=0.5,
                 density=True, label='Original', color='blue')
        plt.hist(synthetic_data[:, :, i].flatten(), bins=50, alpha=0.5,
                 density=True, label='Synthetic', color='red')

        # Add KDE plots
        sns.kdeplot(data=original_data[:, :, i].flatten(), color='blue', linewidth=2)
        sns.kdeplot(data=synthetic_data[:, :, i].flatten(), color='red', linewidth=2)

        plt.title(f'Feature {i} Distribution')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_distributions.png'))
    plt.close()


def _plot_temporal_patterns(original_data, synthetic_data, save_dir):
    """Plot temporal patterns and autocorrelations."""
    n_features = original_data.shape[2]
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols * 2  # *2 for time series and autocorr

    plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    for i in range(n_features):
        # Time series plot
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(original_data[0, :, i], label='Original', color='blue', alpha=0.7)
        plt.plot(synthetic_data[0, :, i], label='Synthetic', color='red', alpha=0.7)
        plt.title(f'Feature {i} Time Series Example')
        plt.legend()

        # Autocorrelation plot
        plt.subplot(n_rows, n_cols, i + 1 + n_features)
        orig_autocorr = np.correlate(original_data[0, :, i],
                                     original_data[0, :, i], mode='full')
        syn_autocorr = np.correlate(synthetic_data[0, :, i],
                                    synthetic_data[0, :, i], mode='full')

        max_lag = min(50, len(orig_autocorr) // 2)
        lags = range(max_lag)
        plt.plot(lags, orig_autocorr[len(orig_autocorr) // 2:len(orig_autocorr) // 2 + max_lag],
                 label='Original', color='blue')
        plt.plot(lags, syn_autocorr[len(syn_autocorr) // 2:len(syn_autocorr) // 2 + max_lag],
                 label='Synthetic', color='red')
        plt.title(f'Feature {i} Autocorrelation')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'temporal_patterns.png'))
    plt.close()


def _plot_correlation_matrices(original_data, synthetic_data, save_dir):
    """Plot and compare correlation matrices."""
    # Calculate correlation matrices
    orig_corr = np.corrcoef(original_data.reshape(-1, original_data.shape[2]).T)
    syn_corr = np.corrcoef(synthetic_data.reshape(-1, synthetic_data.shape[2]).T)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(orig_corr, ax=ax1, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                annot=True, fmt='.2f', square=True)
    ax1.set_title('Original Data Correlation Matrix')

    sns.heatmap(syn_corr, ax=ax2, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                annot=True, fmt='.2f', square=True)
    ax2.set_title('Synthetic Data Correlation Matrix')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_matrices.png'))
    plt.close()


def _plot_pca_analysis(original_data, synthetic_data, save_dir):
    """Perform and visualize PCA analysis."""
    # Reshape data for PCA
    orig_reshaped = original_data.reshape(-1, original_data.shape[2])
    syn_reshaped = synthetic_data.reshape(-1, synthetic_data.shape[2])

    # Combine data for PCA
    combined_data = np.vstack([orig_reshaped, syn_reshaped])

    # Perform PCA
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined_data)

    # Split back into original and synthetic
    n_orig = orig_reshaped.shape[0]
    orig_pca = combined_pca[:n_orig]
    syn_pca = combined_pca[n_orig:]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(orig_pca[:, 0], orig_pca[:, 1], alpha=0.5, label='Original', color='blue')
    plt.scatter(syn_pca[:, 0], syn_pca[:, 1], alpha=0.5, label='Synthetic', color='red')
    plt.title('PCA Analysis of Original vs Synthetic Data')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pca_analysis.png'))
    plt.close()

def train_time_gan(X_minority, device, method_dir, params):
    """
    Train TimeGAN and generate synthetic data.

    Args:
        X_minority: Minority class samples to learn from
        device: torch device
        method_dir: Directory to save results_old
        params: GAN parameters

    Returns:
        synthetic_data: Generated synthetic samples
    """
    print("Training TimeGAN...")

    # Convert data format if needed
    ori_data = np.asarray(X_minority)

    # TimeGAN parameters
    parameters = {
        'module': params.get('module', 'gru'),
        'hidden_dim': params.get('hidden_dim', 24),
        'num_layer': params.get('num_layer', 3),
        'iterations': params.get('iterations', 1000),
        'batch_size': params.get('batch_size', 128),
        'metric_iterations': params.get('metric_iterations', 10)
    }

    # Save initial parameters
    with open(os.path.join(method_dir, 'timegan_params.txt'), 'w') as f:
        for key, value in parameters.items():
            f.write(f"{key}: {value}\n")

    # Train TimeGAN
    print("Original data shape:", ori_data.shape)
    generated_data = timegan(ori_data, parameters)

    return generated_data




def generate_balanced_data_with_gan(X_train_scaled, Y_train, window_weight_train, method_dir, device):
    """
    Generate synthetic data using TimeGAN to balance the dataset.

    Args:
        X_train_scaled: Scaled training data
        Y_train: Training labels
        window_weight_train: Window weights for training data
        method_dir: Directory to save validation plots
        device: torch device

    Returns:
        tuple: (balanced_X, balanced_Y, balanced_weights)
    """
    print("Starting TimeGAN data generation process...")

    # Find minority class samples
    minority_indices = np.where(Y_train == 1)[0]
    majority_indices = np.where(Y_train == 0)[0]
    X_minority = X_train_scaled[minority_indices]

    # Calculate how many synthetic samples needed
    num_synthetic_needed = len(majority_indices) - len(minority_indices)
    print(f"Need to generate {num_synthetic_needed} synthetic samples")

    # TimeGAN parameters
    gan_params = {
        'module': 'gru',
        'hidden_dim': 24,
        'num_layer': 3,
        'iterations': 1000,  # You might want to adjust this
        'batch_size': 128,
        'metric_iterations': 10
    }

    # Train GAN and generate synthetic data
    synthetic_data = train_time_gan(X_minority, device, method_dir, gan_params)

    # Trim synthetic data if needed
    if len(synthetic_data) > num_synthetic_needed:
        synthetic_data = synthetic_data[:num_synthetic_needed]
    elif len(synthetic_data) < num_synthetic_needed:
        print(f"Warning: Generated only {len(synthetic_data)} samples out of {num_synthetic_needed} needed")

    # Combine original and synthetic data
    X_balanced = np.concatenate([X_train_scaled, synthetic_data])
    Y_balanced = np.concatenate([Y_train, np.ones(len(synthetic_data))])
    weights_balanced = np.concatenate([window_weight_train, np.ones(len(synthetic_data))])

    # Print class distribution
    print("\nClass distribution:")
    print(f"Original - {np.bincount(Y_train)}")
    print(f"Balanced - {np.bincount(Y_balanced)}")

    # Validate synthetic data
    gan_validation_dir = os.path.join(method_dir, 'gan_validation')
    os.makedirs(gan_validation_dir, exist_ok=True)
    validate_synthetic_data(X_minority, synthetic_data, gan_validation_dir)

    return X_balanced, Y_balanced, weights_balanced


def main_with_autoencoder(df, window_size=5, method='', resample=False, classification_epochs=20, batch_size=32,
                          ae_epochs=100,
                          depth=4, num_filters=32, lr=0.001, mask_probability=0.4, early_stopping_patience=30,
                          threshold=0.5,
                          use_gan=False):
    seed_everything(0)
    interval = '30ms'

    method_dir = os.path.join('results_old', method)
    os.makedirs(method_dir, exist_ok=True)

    hyperparameters = {
        'window_size': window_size,
        'method': method,
        'resample': resample,
        'classification_epochs': classification_epochs,
        'batch_size': batch_size,
        'ae_epochs': ae_epochs,
        'in_channels': len(feature_columns),
        'num_filters': num_filters,
        'depth': depth,
        'learning_rate': lr,
        'weight_decay': 1e-4,
        'optimizer': 'AdamW',
        'scheduler': 'ReduceLROnPlateau',
        'mask_probability': mask_probability,
        'early_stopping_patience': early_stopping_patience,
        'use_gan': use_gan
    }

    print(hyperparameters)
    with open(os.path.join(method_dir, 'hyperparameters.txt'), 'w') as f:
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")

    # 1. Initial Data Split
    train_df, test_df = cnn_time_series.split_train_test(
        time_series_df=df,
        input_data_points=feature_columns,
        test_size=0.2,
        random_state=0
    )
    print("Original class distribution in test set:")
    print(test_df["target"].value_counts())

    # 2. Create Time Series
    X_train_full, Y_train_full, window_weight_train_full = create_time_series(
        train_df, interval, window_size=window_size, resample=resample)
    X_test, Y_test, window_weight_test = create_time_series(
        test_df, interval, window_size=window_size, resample=resample)

    # 3. Split Training/Validation
    X_train, X_val, Y_train, Y_val, window_weight_train, window_weight_val = train_test_split(
        X_train_full, Y_train_full, window_weight_train_full, test_size=0.2, random_state=42, stratify=Y_train_full)

    # 4. Scale Data
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_reshaped)

    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # 5. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 6. Initialize and Train Autoencoder
    in_channels = len(feature_columns)
    input_length = X_train.shape[1]
    hidden_size = 128
    num_layers = 1
    rnn_type = 'GRU'

    autoencoder = CNNRecurrentAutoencoder(
        in_channels=in_channels,
        num_filters=num_filters,
        depth=depth,
        hidden_size=hidden_size,
        num_layers=num_layers,
        rnn_type=rnn_type,
        input_length=input_length
    ).to(device)

    autoencoder.apply(initialize_weights)
    best_autoencoder_path = os.path.join(method_dir, 'best_autoencoder.pth')

    # Train or load autoencoder
    if TRAIN:
        criterion_ae = nn.MSELoss()
        optimizer_ae = optim.AdamW(autoencoder.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler_ae = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ae, mode='min', factor=0.5, patience=20)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).permute(0, 2, 1)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
        Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, Y_val_tensor)

        train_loader_ae = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader_ae = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        train_autoencoder_with_input_masking(
            autoencoder,
            train_loader_ae,
            val_loader_ae,
            criterion_ae,
            optimizer_ae,
            epochs=ae_epochs,
            device=device,
            scheduler=scheduler_ae,
            mask_probability=mask_probability,
            save_path=method_dir,
            early_stopping_patience=early_stopping_patience
        )
        torch.save(autoencoder.state_dict(), best_autoencoder_path)

    autoencoder.load_state_dict(torch.load(best_autoencoder_path))

    # 7. Generate Synthetic Data if use_gan is True
    if use_gan:
        print("\nGenerating synthetic data using TimeGAN...")
        X_train_balanced, Y_train_balanced, window_weight_balanced = generate_balanced_data_with_gan(
            X_train_scaled, Y_train, window_weight_train, method_dir, device)
        print("Finished generating synthetic data")
    else:
        X_train_balanced = X_train_scaled
        Y_train_balanced = Y_train
        window_weight_balanced = window_weight_train

    # 8. Prepare Data Loaders for Classification
    X_train_tensor = torch.tensor(X_train_balanced, dtype=torch.float32).permute(0, 2, 1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).permute(0, 2, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).permute(0, 2, 1)

    Y_train_tensor = torch.tensor(Y_train_balanced, dtype=torch.long)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, Y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor)

    if use_gan:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        # Use weighted sampling for imbalanced data
        train_labels_numpy = Y_train_balanced
        classes = np.unique(train_labels_numpy)
        class_weights = compute_class_weight('balanced', classes=classes, y=train_labels_numpy)
        samples_weight = np.array([class_weights[t] for t in train_labels_numpy])
        samples_weight = torch.from_numpy(samples_weight).float()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 9. Initialize Combined Model
    num_classes = 2
    model = CombinedModel(autoencoder, num_classes).to(device)

    # 10. Setup Optimizer with Different Learning Rates
    encoder_params = list(model.encoder.parameters())
    classifier_params = list(model.classifier.parameters())
    optimizer = optim.AdamW([
        {'params': encoder_params, 'lr': lr * 0.1},
        {'params': classifier_params, 'lr': lr}
    ], weight_decay=1e-4)

    # 11. Setup Loss and Scheduler
    window_weight_train_tensor = torch.tensor(window_weight_balanced, dtype=torch.float32).to(device)
    window_weight_val_tensor = torch.tensor(window_weight_val, dtype=torch.float32).to(device)
    criterion = WeightedCrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 12. Train Combined Model
    train_combined_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        classification_epochs,
        device,
        scheduler,
        save_path=method_dir,
        early_stopping_patience=5,
        window_weight_train_tensor=window_weight_train_tensor,
        window_weight_val_tensor=window_weight_val_tensor
    )

    results_text_path = os.path.join(method_dir, 'result.txt')
    evaluate_model(model, test_loader, device, results_text_path, threshold)


if __name__ == '__main__':
    # Load your dataset
    categorized_rad_s1_s18_file_path = 'data/Categorized_Fixation_Data_1_18.csv'
    approach_num = 6
    categorized_rad_init_kwargs = dict(
        data_file_path=categorized_rad_s1_s18_file_path,
        test_data_file_path=None,
        augment=False,
        join_train_and_test_data=False,
        normalize=True,
        remove_surrounding_to_hits=0,
        update_surrounding_to_hits=0,
        approach_num=approach_num,
    )

    ident_sub_rec = IdentSubRec(**categorized_rad_init_kwargs)
    df = ident_sub_rec.df
    window = 50
    classification_epochs = 10
    batch_size = 32
    ae_epochs = 100
    depth = 4
    num_filters = 32
    lr = 0.001
    mask_probability = 0.4
    participant = 1
    threshold = 0.9
    method = f" Add GRU to encoder decoder , approach_num ={approach_num},depth = {depth},lr={lr},ae_epochs = {ae_epochs},mask_prob ={mask_probability}" \
             f"num_filters = {num_filters},participant = {1}"
    new_df = df[df['RECORDING_SESSION_LABEL'] == participant]

    main_with_autoencoder(new_df, window_size=window, method=method, classification_epochs=classification_epochs, batch_size=batch_size,
                              ae_epochs=ae_epochs, depth=depth, num_filters=num_filters, lr=lr,
                              mask_probability=mask_probability,threshold =threshold)



    # main_with_autoencoder(df, window_size=window, method=method,epochs=epochs,batch_size=batch_size,ae_epochs=ae_epochs,depth=depth,num_filters=num_filters,lr=lr,mask_probability=mask_probability)
    # labels = list(df['RECORDING_SESSION_LABEL'].unique())
    # for label in labels:
    #     print(f"############################LABEL {label}#########################")
    #     method = f" non independent Inception Auto encoder, approach_num ={approach_num},depth = {depth},lr={lr},ae_epochs = {ae_epochs},mask_prob ={mask_probability}" \
    #              f"num_filters = {num_filters},ID = {label}"
    #     new_df = df[df['RECORDING_SESSION_LABEL'] == label]
    #     main_with_autoencoder(new_df, window_size=window, method=method, epochs=epochs, batch_size=batch_size,
    #                           ae_epochs=ae_epochs, depth=depth, num_filters=num_filters, lr=lr,
    #                           mask_probability=mask_probability)
    #
# Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.98      0.98      0.98     11978
#            1       0.03      0.03      0.03       187
#
#     accuracy                           0.97     12165
#    macro avg       0.51      0.51      0.51     12165
# weighted avg       0.97      0.97      0.97     12165
#
# Confusion Matrix:
#    Predicted 0  Predicted 1
# Actual 0   11795      183
# Actual 1   181        6

# Classification Report:
#               precision    recall  f1-score   support
#
#            0       0.86      0.69      0.76     10269
#            1       0.13      0.30      0.18      1626
#
#     accuracy                           0.63     11895
#    macro avg       0.50      0.49      0.47     11895
# weighted avg       0.76      0.63      0.68     11895
#
# Confusion Matrix:
#    Predicted 0  Predicted 1
# Actual 0   7050       3219
# Actual 1   1133       493



# Classification Report: threshold 0.9
#               precision    recall  f1-score   support
#            0       0.88      0.84      0.86     10269
#            1       0.22      0.28      0.25      1626
#     accuracy                           0.76     11895
#    macro avg       0.55      0.56      0.55     11895
# weighted avg       0.79      0.76      0.78     11895
# Confusion Matrix:
#    Predicted 0  Predicted 1
# Actual 0   8640       1629
# Actual 1   1169       457