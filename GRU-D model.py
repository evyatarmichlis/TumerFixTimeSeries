import os
import random

import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.utils import class_weight

from data_process import IdentSubRec

from sklearn.model_selection import GroupShuffleSplit

import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

TRAIN = True

def seed_everything(seed):
    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
def split_train_test(time_series_df, input_data_points, test_size=0.2, random_state=0):
    df = time_series_df
    df['group'] = df[['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1
    )
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    split_indices = list(gss.split(X=df[input_data_points], y=df['target'], groups=df['group']))[0]

    train_index, test_index = split_indices

    x_train = df.iloc[train_index].drop(columns='target', errors='ignore')
    x_test = df.iloc[test_index].drop(columns='target', errors='ignore')

    y_train = df['target'].iloc[train_index]
    y_test = df['target'].iloc[test_index]

    train_df = pd.concat([x_train, y_train], axis=1)
    test_df = pd.concat([x_test, y_test], axis=1)

    return train_df, test_df

feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X', 'CURRENT_FIX_IA_Y',
                   'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT']

input_data_points = [
    'RECORDING_SESSION_LABEL',
    'TRIAL_INDEX',
    'CURRENT_FIX_INDEX',
    'Pupil_Size',
    'CURRENT_FIX_DURATION',
    'CURRENT_FIX_IA_X',
    'CURRENT_FIX_IA_Y',
    'CURRENT_FIX_COMPONENT_COUNT',
    'CURRENT_FIX_COMPONENT_INDEX',
    'CURRENT_FIX_COMPONENT_DURATION',
]

def create_windows(grouped, window_size):
    max_possible_time_length = 0

    for name, group in grouped:
        group = group.sort_values(by='Cumulative_Time')
        for start in range(0, len(group) - window_size + 1):
            window = group.iloc[start:start + window_size]
            time_length = window['Cumulative_Time'].max() - window['Cumulative_Time'].min()
            time_length_seconds = time_length.total_seconds()
            if time_length_seconds > max_possible_time_length:
                max_possible_time_length = time_length_seconds

    samples = []
    labels = []
    weights = []

    for name, group in grouped:
        group = group.sort_values(by='Cumulative_Time')
        for start in range(0, len(group) - window_size + 1):
            window = group.iloc[start:start + window_size]
            features = window[feature_columns].values
            samples.append(features)

            # Calculate the label
            label = window['target'].max()
            labels.append(label)

            # Calculate the time length of the window
            time_length_seconds = window['Cumulative_Time'].max().total_seconds() - window[
                'Cumulative_Time'].min().total_seconds()

            # Normalize the weight
            normalized_weight = time_length_seconds / max_possible_time_length
            weights.append(normalized_weight)

    samples = np.array(samples)
    labels = np.array(labels)
    weights = np.array(weights)
    return samples, labels, weights

def resample_func(time_series_df, interval):
    resampled_data = []

    for (session, trial), group in time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX', 'CURRENT_FIX_INDEX']):
        group = group.set_index('Cumulative_Time')

        # Resample features to a fixed interval (e.g., 1 second)
        resampled_features = group[feature_columns].resample(interval).mean()

        # Resample the target column using the max function
        resampled_target = group['target'].resample(interval).max()

        resampled_group = resampled_features.merge(resampled_target, on='Cumulative_Time', how='left')

        resampled_group.ffill(inplace=True)
        resampled_group.bfill(inplace=True)

        # Add 'RECORDING_SESSION_LABEL' and 'TRIAL_INDEX' back to the resampled group
        resampled_group['RECORDING_SESSION_LABEL'] = session
        resampled_group['TRIAL_INDEX'] = trial

        # Append the resampled group to the list
        resampled_data.append(resampled_group)

    # Concatenate all resampled groups into a single DataFrame
    time_series_df = pd.concat(resampled_data).reset_index()

    return time_series_df

def create_time_series(time_series_df, interval='30ms', window_size=5, resample=False):
    time_series_df = time_series_df.copy()

    time_series_df['Cumulative_Time_Update'] = (time_series_df['CURRENT_FIX_COMPONENT_INDEX'] == 1).astype(int)
    time_series_df['Cumulative_Time'] = 0.0  # Ensure float type for cumulative times

    # Iterate over each group and calculate cumulative time correctly
    for _, group in time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']):
        cumulative_time = 0
        cumulative_times = []
        for index, row in group.iterrows():
            if row['Cumulative_Time_Update'] == 1:
                cumulative_time += row['CURRENT_FIX_DURATION']
            cumulative_times.append(cumulative_time)
        time_series_df.loc[group.index, 'Cumulative_Time'] = cumulative_times

    time_series_df['Unique_Index'] = time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']).cumcount()
    time_series_df['Cumulative_Time'] += time_series_df['Unique_Index'] * 1e-4

    # Set 'Cumulative_Time' as a timedelta for easier resampling
    time_series_df['Cumulative_Time'] = pd.to_timedelta(time_series_df['Cumulative_Time'], unit='s')

    if resample:
        time_series_df = resample_func(time_series_df, interval)

    grouped = time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'])

    return create_windows(grouped, window_size)

# ------------------ GRU-d Model Definition ------------------

class GRUDModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(GRUDModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define GRU layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

        # Define a fully connected output layer

    def forward(self, x):
        # Determine the number of directions
        num_directions = 2 if self.gru.bidirectional else 1

        # Initialize hidden state with correct dimensions
        h0 = torch.zeros(self.num_layers * num_directions, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate GRU
        out, _ = self.gru(x, h0)

        # Take the output from the last time step
        out = out[:, -1, :]

        out = self.fc(out)
        return out

# ------------------------------------------------------------

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

def get_class_weights(Y_train):
    """Compute class weights for imbalanced datasets."""
    class_weights = compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)
    return {i: class_weights[i] for i in range(len(class_weights))}

def initialize_model(input_shape, num_classes, class_weights):
    """Initialize the model, criterion, and optimizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize GRU-d model
    model = GRUDModel(input_size=input_shape[2], hidden_size=64, output_size=num_classes).to(device)
    criterion = FocalLoss(gamma=2., alpha=class_weights[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    return device, model, criterion, optimizer

def prepare_data_loaders(X_train, Y_train, window_weight_train, batch_size, device):
    """Prepare training and validation data loaders."""
    X_train, X_val, Y_train, Y_val, window_weight_train, window_weight_val = train_test_split(
        X_train, Y_train, window_weight_train, test_size=0.2, random_state=42
    )

    # Convert datasets to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long).to(device)

    # Compute sample weights
    class_sample_count = np.array([len(np.where(Y_train_tensor.cpu() == t)[0]) for t in np.unique(Y_train_tensor.cpu())])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in Y_train_tensor.cpu()])
    samples_weight = torch.from_numpy(samples_weight).to(device)
    window_weight_train = torch.tensor(window_weight_train, dtype=torch.float32).to(device)
    combined_weights = samples_weight * window_weight_train

    sampler = WeightedRandomSampler(combined_weights, len(combined_weights))

    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, Y_val_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def validate_model(model, val_loader, criterion, epoch, running_loss, train_loader, device):
    """Validate the model during training."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}")

def print_metrics(precision_weighted, recall_weighted, f1_weighted, precision_macro, recall_macro, f1_macro):
    """Print evaluation metrics."""
    print(f"Weighted Precision: {precision_weighted}")
    print(f"Weighted Recall: {recall_weighted}")
    print(f"Weighted F1-score: {f1_weighted}")

    print(f"Macro Precision: {precision_macro}")
    print(f"Macro Recall: {recall_macro}")
    print(f"Macro F1-score: {f1_macro}")

def evaluate_and_save_model(model, X_test, Y_test, device, method):
    """Evaluate the model and save results_old."""
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        outputs = model(X_test)
        _, y_pred_classes = torch.max(outputs, 1)

    # Evaluate metrics
    precision_weighted = precision_score(Y_test, y_pred_classes.cpu(), average='weighted')
    recall_weighted = recall_score(Y_test, y_pred_classes.cpu(), average='weighted')
    f1_weighted = f1_score(Y_test, y_pred_classes.cpu(), average='weighted')

    precision_macro = precision_score(Y_test, y_pred_classes.cpu(), average='macro')
    recall_macro = recall_score(Y_test, y_pred_classes.cpu(), average='macro')
    f1_macro = f1_score(Y_test, y_pred_classes.cpu(), average='macro')

    print_metrics(precision_weighted, recall_weighted, f1_weighted, precision_macro, recall_macro, f1_macro)
    print(classification_report(Y_test, y_pred_classes.cpu()))

    plot_confusion_matrix(Y_test, y_pred_classes.cpu())

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    """Train the model and evaluate on the validation set."""
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch % 10 == 0:
            validate_model(model, val_loader, criterion, epoch, running_loss, train_loader, device)

def plot_confusion_matrix(y_true, y_pred, save_path=''):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    # Print the confusion matrix in a readable format
    print("Confusion Matrix:")
    print("   Predicted 0  Predicted 1")
    print(f"Actual 0   {cm[0, 0]:<10} {cm[0, 1]:<10}")
    print(f"Actual 1   {cm[1, 0]:<10} {cm[1, 1]:<10}")

def main(df, window_size=5, method='', resample=False, epochs=100, batch_size=32):
    interval = '30ms'

    # Step 1: Split the dataset
    train_df, test_df = split_train_test(
        time_series_df=df,
        input_data_points=feature_columns,
        test_size=0.2,
        random_state=0
    )

    # Step 2: Create time series data for training and testing
    X_train, Y_train, window_weight_train = create_time_series(train_df, interval, window_size=window_size, resample=resample)
    X_test, Y_test, window_weights_test = create_time_series(test_df, interval, window_size=window_size, resample=resample)

    # Step 3: Compute class weights for handling class imbalance
    class_weights_dict = get_class_weights(Y_train)
    print(class_weights_dict)

    # Step 4: Initialize model, loss, optimizer, and device
    device, model, criterion, optimizer = initialize_model(
        input_shape=X_train.shape,
        num_classes=2,
        class_weights=class_weights_dict
    )

    # Step 5: Prepare data loaders
    train_loader, val_loader = prepare_data_loaders(X_train, Y_train, window_weight_train, batch_size=batch_size, device=device)

    # Step 6: Train the model
    if TRAIN:
        train_model(model, train_loader, val_loader, criterion, optimizer, epochs=epochs, device=device)
        torch.save(model.state_dict(), f'gru_d_model/best_gru_d_{method}.pth')

    # Step 7: Save and evaluate the model
    model.load_state_dict(torch.load(f'gru_d_model/best_gru_d_{method}.pth'))

    evaluate_and_save_model(model, X_test, Y_test, device, method)

if __name__ == '__main__':
    categorized_rad_s1_s18_file_path = 'data/Categorized_Fixation_Data_1_18.csv'

    raw_participants_file_paths = ['1_Formatted_Sample.csv', '2_Formatted_Sample.csv', '3_Formatted_Sample.csv',
                                   '4_Formatted_Sample.csv', '6_Formatted_Sample.csv', '7_Formatted_Sample.csv',
                                   '8_Formatted_Sample.csv', '9_Formatted_Sample.csv', '19_Formatted_Sample.csv',
                                   '23_Formatted_Sample.csv', '24_Formatted_Sample.csv', '25_Formatted_Sample.csv',
                                   '30_Formatted_Sample.csv', '33_Formatted_Sample.csv', '34_Formatted_Sample.csv',
                                   '35_Formatted_Sample.csv', '36_Formatted_Sample.csv', '37_Formatted_Sample.csv']
    raw_participants_folder_path = '/media/y/2TB/1_THESIS_FILES/Data_D231110/'
    for i in range(len(raw_participants_file_paths)):
        raw_participants_file_paths[i] = os.path.join(raw_participants_folder_path, raw_participants_file_paths[i])

    # Hit in the categorized data are 2 for nodule, 1 for surrounding and 0 for non-nodule.
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
    seed_everything(0)
    labels = list(df['RECORDING_SESSION_LABEL'].unique())
    for label in labels[:1]:
        print(f"############################LABEL {label}#########################")

        new_df = df[df['RECORDING_SESSION_LABEL'] == label]
        main(new_df, window_size=50, method=f"test_all_ids_{label}", resample=False)
