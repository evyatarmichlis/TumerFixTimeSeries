import os

import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import torch

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score,confusion_matrix
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

def split_train_test(time_series_df):
    df =time_series_df
    df['group'] = df[['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1)
    gss = GroupShuffleSplit(n_splits=1,
                            test_size=0.2,
                            random_state=42)

    split_indices = list(gss.split(X=df[input_data_points],
                                   y=df['target'],
                                   groups=df['group']))[0]
    train_index, test_index = split_indices
    x_train, x_test = df[input_data_points].iloc[train_index], df[input_data_points].iloc[test_index]
    y_train, y_test = df['target'].iloc[train_index], df['target'].iloc[test_index]
    train_df = pd.concat([x_train, y_train], axis=1)
    test_df = pd.concat([x_test, y_test], axis=1)

    return train_df, test_df

def create_windows(grouped,window_size):
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
    return samples,labels,weights
def resample_func(time_series_df,interval):
    resampled_data = []

    for (session, trial), group in time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']):
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
def create_time_series(time_series_df, interval='30ms',  window_size = 5 ,resample= False):
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

    if resample :
        time_series_df = resample_func(time_series_df,interval)

    grouped = time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'])



    return create_windows(grouped,window_size)



class DilatedConv1DModel(nn.Module):
    def __init__(self, input_shape, num_classes, dilation_rates=[1, 2, 4]):
        super(DilatedConv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=64, kernel_size=3, padding=1,
                               dilation=dilation_rates[0])
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2, dilation=dilation_rates[1])
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=4, dilation=dilation_rates[2])
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x
class Conv1DModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(Conv1DModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x


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

    model = DilatedConv1DModel(input_shape=input_shape, num_classes=num_classes).to(device)
    criterion = FocalLoss(gamma=2., alpha=class_weights[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    return device, model, criterion, optimizer


def prepare_data_loaders(X_train, Y_train, window_weight_train, batch_size, device):
    """Prepare training and validation data loaders."""
    X_train, X_val, Y_train, Y_val, window_weight_train, window_weight_val = train_test_split(
        X_train, Y_train, window_weight_train, test_size=0.2, random_state=42
    )

    # Convert datasets to torch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1).to(device)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1).to(device)
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
    """Evaluate the model and save results."""
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)
        outputs = model(X_test)
        _, y_pred_classes = torch.max(outputs, 1)

    # Save the model
    torch.save(model.state_dict(), f'{method}_model.pth')

    # Evaluate metrics
    precision_weighted = precision_score(Y_test, y_pred_classes, average='weighted')
    recall_weighted = recall_score(Y_test, y_pred_classes, average='weighted')
    f1_weighted = f1_score(Y_test, y_pred_classes, average='weighted')

    precision_macro = precision_score(Y_test, y_pred_classes, average='macro')
    recall_macro = recall_score(Y_test, y_pred_classes, average='macro')
    f1_macro = f1_score(Y_test, y_pred_classes, average='macro')

    print_metrics(precision_weighted, recall_weighted, f1_weighted, precision_macro, recall_macro, f1_macro)

    plot_confusion_matrix(Y_test, y_pred_classes)


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


def plot_confusion_matrix(y_true, y_pred,save_path=''):
    """Plots and saves the confusion matrix."""
    # Compute confusion matrix
    confusion_matrix_res = confusion_matrix(y_true, y_pred)

    # Create a figure for the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix_res, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])

    # Add labels and title
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Save and show the confusion matrix
    plt.savefig(f'{save_path}/confusion_matrix.png')
    plt.show()
def main(df, window_size=5, method='', resample=False,epochs=200,batch_size=32):
    interval = '10ms'

    # Step 1: Split the dataset
    train_df, test_df = split_train_test(df)

    # Step 2: Create time series data for training and testing
    X_train, Y_train, window_weight_train = create_time_series(train_df, interval, window_size=window_size, resample=resample)
    X_test, Y_test, window_weights_test = create_time_series(test_df, interval, window_size=window_size, resample=resample)

    # Step 3: Compute class weights for handling class imbalance
    class_weights_dict = get_class_weights(Y_train)
    print(class_weights_dict)

    # Step 4: Initialize model, loss, optimizer, and device
    device, model, criterion, optimizer = initialize_model(input_shape=(window_size, len(feature_columns)),
                                                           num_classes=2,
                                                           class_weights=class_weights_dict)

    # Step 5: Prepare data loaders
    train_loader, val_loader = prepare_data_loaders(X_train, Y_train, window_weight_train, batch_size=batch_size, device=device)

    # Step 6: Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=epochs, device=device)

    # Step 7: Save and evaluate the model
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
    for window in [5,10,20,50,100,500]:
        print(window)
        main(df, window_size=window, method=f'new train_test')
    # main(df,window_size=window,method='add weighted random sampler and dialconv')
    # labels = list(df['RECORDING_SESSION_LABEL'].unique())
    #
    #
    # for label in labels[1:]:
    #     print(label)


    #first results without adding class balance to the model:
    #Weighted Precision: 0.9716223333336166
    # Weighted Recall: 0.9608698687517453
    # Weighted F1-score: 0.9662139812876347
    # Macro Precision: 0.4934640121992207
    # Macro Recall: 0.4885411571445869
    # Macro F1-score: 0.49091236859086845
    # Micro Precision: 0.9608698687517453
    # Micro Recall: 0.9608698687517453
    # Micro F1-score: 0.9608698687517453

    #second try withadding class balance
    # window size = 5
    # Weighted Precision: 0.9438468040422491
    # Weighted Recall: 0.9145134025597682
    # Weighted F1-score: 0.9287729082501983
    # Macro Precision: 0.5002833273614268
    # Macro Recall: 0.5005694541857866
    # Macro F1-score: 0.4975241883515054
    # Micro Precision: 0.9145134025597682
    # Micro Recall: 0.9145134025597682
    # Micro F1-score: 0.9145134025597682


    # window size = 10 - 100 epochs
    #Weighted Precision: 0.9137752414547997
    # Weighted Recall: 0.8722398402630961
    # Weighted F1-score: 0.8921599296687734
    # Macro Precision: 0.4981799005443969
    # Macro Recall: 0.4965104806123435
    # Macro F1-score: 0.49352666214315266
    # Micro Precision: 0.8722398402630961
    # Micro Recall: 0.8722398402630961
    # Micro F1-score: 0.8722398402630961


    # window size = 15 - 100 epochs
    # Weighted Precision: 0.9275606782876191
    # Weighted Recall: 0.8449985964255637
    # Weighted F1-score: 0.882882959576623
    # Macro Precision: 0.506984739830627
    # Macro Recall: 0.5210768306872349
    # Macro F1-score: 0.49691276052753636
    # Micro Precision: 0.8449985964255637
    # Micro Recall: 0.8449985964255637
    # Micro F1-score: 0.8449985964255637

    # window size = 20 - 100 epochs

    # Weighted
    # Precision: 0.9135915228311815
    # Weighted
    # Recall: 0.8501689685571554
    # Weighted
    # F1 - score: 0.8799632770918373
    # Macro
    # Precision: 0.4991192736863065
    # Macro
    # Recall: 0.4979234465036716
    # Macro
    # F1 - score: 0.4905461616369552
    # Micro
    # Precision: 0.8501689685571554
    # Micro
    # Recall: 0.8501689685571554
    # Micro
    # F1 - score: 0.8501689685571554
    #

    # window size = 30 - 100 epochs

    #     Weighted Precision: 0.8536910631896126
    # Weighted Recall: 0.8645426350344383
    # Weighted F1-score: 0.8590591644341848
    # Macro Precision: 0.4794185713153696
    # Macro Recall: 0.48250325278388895
    # Macro F1-score: 0.480807564569495
    # Micro Precision: 0.8645426350344383
    # Micro Recall: 0.8645426350344383
# Micro F1-score: 0.8645426350344383





# test with f1 as loss- window - 15
# Weighted Precision: 0.9496710234810042
# Weighted Recall: 0.8972477791878173
# Weighted F1-score: 0.922542137528438
# Macro Precision: 0.4923708072395619
# Macro Recall: 0.4771235150084106
# Macro F1-score: 0.48124493454866185
# Micro Precision: 0.8972477791878173
# Micro Recall: 0.8972477791878173
# Micro F1-score: 0.8972477791878173

#res with lstm model
# Weighted Precision: 0.9596333843520011
# Weighted Recall: 0.6981725351016469
# Weighted F1-score: 0.8052638723958315
# Macro Precision: 0.4986145834057612
# Macro Recall: 0.48550146157023477
# Macro F1-score: 0.4276079437728776
# Micro Precision: 0.6981725351016469
# Micro Recall: 0.6981725351016469
# Micro F1-score: 0.6981725351016469


#only minorty
# Weighted Precision: 0.9922589351490567
# Weighted Recall: 0.8769739810497819
# Weighted F1-score: 0.9309680700207712
# Macro Precision: 0.4987382254430667
# Macro Recall: 0.46423159071673653
# Macro F1-score: 0.468683961620218
# Micro Precision: 0.876973981049782
# Micro Recall: 0.876973981049782
# Micro F1-score: 0.876973981049782


#oversampling
# Weighted Precision: 0.9803214206444
# Weighted Recall: 0.9691575935491064
# Weighted F1-score: 0.9747037935719062
# Macro Precision: 0.49588844325108683
# Macro Recall: 0.4912548430603934
# Macro F1-score: 0.4933702567247812
# Micro Precision: 0.9691575935491064
# Micro Recall: 0.9691575935491064
# Micro F1-score: 0.9691575935491064


#focal loss and weights
# Weighted Precision: 0.9085334141025153
# Weighted Recall: 0.8455248990578735
# Weighted F1-score: 0.8750696577995345
# Macro Precision: 0.4998451559689207
# Macro Recall: 0.49964821516441565
# Macro F1-score: 0.49180095265541873
# Micro Precision: 0.8455248990578735
# Micro Recall: 0.8455248990578735
# Micro F1-score: 0.8455248990578735


#focal loss without weights

# Weighted Precision: 0.9708910123654156
# Weighted Recall: 0.9794111591517397
# Weighted F1-score: 0.9751324750780583
# Macro Precision: 0.49264705882352944
# Macro Recall: 0.4969703301295445
# Macro F1-score: 0.4947992510921573
# Micro Precision: 0.9794111591517397
# Micro Recall: 0.9794111591517397
# Micro F1-score: 0.9794111591517397



#compre to yakir- with his test train split. - method 7
# Weighted Precision: 0.9885544302005926
# Weighted Recall: 0.9884785252512321
# Weighted F1-score: 0.988516303496793
# Macro Precision: 0.6977815454381543
# Macro Recall: 0.7003927271160516
# Macro F1-score: 0.6990782751323477
# Micro Precision: 0.9884785252512321
# Micro Recall: 0.9884785252512321
# Micro F1-score: 0.9884785252512321


#compre to yakir- with his test train split. - method 6
# Weighted Precision: 0.9390139439640396
# Weighted Recall: 0.9347520874898396
# Weighted F1-score: 0.9042669516644848
# Macro Precision: 0.9673410511521249
# Macro Recall: 0.5080779944289694
# Macro F1-score: 0.49901833975280024
# Micro Precision: 0.9347520874898396
# Micro Recall: 0.9347520874898396
# Micro F1-score: 0.9347520874898396



#yakir result- mthod 7
# max results are:
# Accuracy 97.51%
# Precision 52.68%
# Recall 33.52%
# F1 score 40.97%
# ROC AUC score 95.73%

#yakir result- mthod 6
# Accuracy 98.35%
# Precision 67.80%
# Recall 18.26%
# F1 score 28.78%
# ROC AUC score 91.66%


#user 2
# Weighted Precision: 0.9909197761651057
# Weighted Recall: 0.9908390758005675
# Weighted F1-score: 0.9908794243397132
# Macro Precision: 0.4977194982896237
# Macro Recall: 0.4976789640850232
# Macro F1-score: 0.4976992303620149
# Micro Precision: 0.9908390758005675
# Micro Recall: 0.9908390758005675
# Micro F1-score: 0.9908390758005675

#user 3
# Weighted Precision: 0.9633614124058109
# Weighted Recall: 0.9714675501851491
# Weighted F1-score: 0.9648468696963706
# Macro Precision: 0.7628923062788775
# Macro Recall: 0.5957052749186286
# Macro F1-score: 0.6373809578979077
# Micro Precision: 0.9714675501851491
# Micro Recall: 0.9714675501851491
# Micro F1-score: 0.9714675501851491

#user 4
# Weighted Precision: 0.9593232876398184
# Weighted Recall: 0.9694824349691606
# Weighted F1-score: 0.959324713815874
# Macro Precision: 0.7716582775126993
# Macro Recall: 0.5533719510463696
# Macro F1-score: 0.5840473836812327
# Micro Precision: 0.9694824349691606
# Micro Recall: 0.9694824349691606
# Micro F1-score: 0.9694824349691606

#user 5
#Weighted Precision: 0.9766725487187357
# Weighted Recall: 0.9825456498388829
# Weighted F1-score: 0.976898061452974
# Macro Precision: 0.7777628032345013
# Macro Recall: 0.5588811479926047
# Macro F1-score: 0.5943538670112128
# Micro Precision: 0.9825456498388829
# Micro Recall: 0.9825456498388829
# Micro F1-score: 0.9825456498388829

#user 6
# Weighted Precision: 0.9722393888975919
# Weighted Recall: 0.97487851838269
# Weighted F1-score: 0.973382066584111
# Macro Precision: 0.7445045070793364
# Macro Recall: 0.6924313158536901
# Macro F1-score: 0.7151194270126737
# Micro Precision: 0.97487851838269
# Micro Recall: 0.97487851838269
# Micro F1-score: 0.97487851838269



#user 7
# Weighted Precision: 0.9473027581194606
# Weighted Recall: 0.9369603565128586
# Weighted F1-score: 0.9420344416340997
# Macro Precision: 0.5202792901590643
# Macro Recall: 0.5280703376365936
# Macro F1-score: 0.5230305182483728
# Micro Precision: 0.9369603565128586
# Micro Recall: 0.9369603565128586
# Micro F1-score: 0.9369603565128586

#user 8

# Weighted Precision: 0.9802749981120723
# Weighted Recall: 0.9833019755409219
# Weighted F1-score: 0.9817861536884338
# Macro Precision: 0.49502723182571634
# Macro Recall: 0.49655581947743466
# Macro F1-score: 0.495790347444563
# Micro Precision: 0.9833019755409219
# Micro Recall: 0.9833019755409219
# Micro F1-score: 0.9833019755409219

#user 8


# Weighted Precision: 0.9751918211850816
# Weighted Recall: 0.9787880747591023
# Weighted F1-score: 0.9766547142275266
# Macro Precision: 0.7234710344638877
# Macro Recall: 0.6485804651973551
# Macro F1-score: 0.6780043658496697
# Micro Precision: 0.9787880747591023
# Micro Recall: 0.9787880747591023
# Micro F1-score: 0.9787880747591023

#user 9

#Weighted Precision: 0.9599226630345505
# Weighted Recall: 0.9769384324437907
# Weighted F1-score: 0.9678485069471099
# Macro Precision: 0.5228810138135608
# Macro Recall: 0.5026231487443658
# Macro F1-score: 0.5009079960809877
# Micro Precision: 0.9769384324437907
# Micro Recall: 0.9769384324437907
# Micro F1-score: 0.9769384324437907

#user 9

# Weighted Precision: 0.9694422721653925
# Weighted Recall: 0.9828784539752501
# Weighted F1-score: 0.9757234941781425
# Macro Precision: 0.5233141656764556
# Macro Recall: 0.5020135796360438
# Macro F1-score: 0.5005842453260374
# Micro Precision: 0.9828784539752501
# Micro Recall: 0.9828784539752501
# Micro F1-score: 0.9828784539752501
# #user 10
# Weighted Precision: 0.9645959438055145
# Weighted Recall: 0.9711950970377937
# Weighted F1-score: 0.9593334518661358
# Macro Precision: 0.8513022720826899
# Macro Recall: 0.5319443489274396
# Macro F1-score: 0.5520541277258567
# Micro Precision: 0.9711950970377937
# Micro Recall: 0.9711950970377937
# Micro F1-score: 0.9711950970377937
# #user 11
# Weighted Precision: 0.9697953014600845
# Weighted Recall: 0.9845283707352673
# Weighted F1-score: 0.9771063019675429
# Macro Precision: 0.4923899554712023
# Macro Recall: 0.4998703127315844
# Macro F1-score: 0.4961019379987497
# Micro Precision: 0.9845283707352673
# Micro Recall: 0.9845283707352673
# Micro F1-score: 0.9845283707352673
# #user 12
# Weighted Precision: 0.9650149077287252
# Weighted Recall: 0.9817597059499196
# Weighted F1-score: 0.9733152932981813
# Macro Precision: 0.49117322545053327
# Macro Recall: 0.4996959917683925
# Macro F1-score: 0.49539795516194096
# Micro Precision: 0.9817597059499196
# Micro Recall: 0.9817597059499196
# Micro F1-score: 0.9817597059499196
# #user 13
# Weighted Precision: 0.9791840708008908
# Weighted Recall: 0.9800760582010583
# Weighted F1-score: 0.9796296804123911
# Macro Precision: 0.4990515802199067
# Macro Recall: 0.49913293067458386
# Macro F1-score: 0.4990837010128003
# Micro Precision: 0.9800760582010583
# Micro Recall: 0.9800760582010583
# Micro F1-score: 0.9800760582010583
# #user 14
# Weighted Precision: 0.9600174636469071
# Weighted Recall: 0.9601696279993558
# Weighted F1-score: 0.9600934197781064
# Macro Precision: 0.6250847033632061
# Macro Recall: 0.6241310443267765
# Macro F1-score: 0.6246057777833783
# Micro Precision: 0.9601696279993558
# Micro Recall: 0.9601696279993558
# Micro F1-score: 0.9601696279993558
# #user 15
#
# Weighted Precision: 0.9782813755977632
# Weighted Recall: 0.9810993507071066
# Weighted F1-score: 0.9768449300319639
# Macro Precision: 0.8903983442722577
# Macro Recall: 0.647417281897683
# Macro F1-score: 0.7114831926441989
# Micro Precision: 0.9810993507071066
# Micro Recall: 0.9810993507071066
# Micro F1-score: 0.9810993507071066
# #user 16
# Weighted Precision: 0.9805802274714833
# Weighted Recall: 0.9801840822956145
# Weighted F1-score: 0.9750285925147875
# Macro Precision: 0.9900043696744593
# Macro Recall: 0.6534090909090909
# Macro F1-score: 0.7297343328335832
# Micro Precision: 0.9801840822956145
# Micro Recall: 0.9801840822956145
# Micro F1-score: 0.9801840822956145


# yakir results:
# mean results are:
# Accuracy 97.51%
# Precision 52.68%
# Recall 33.52%
# F1 score 40.97%
# ROC AUC score 95.73%
#
#
# max results are:
# Accuracy 97.51%
# Precision 52.68%
# Recall 33.52%
# F1 score 40.97%
# ROC AUC score 95.73%
#
#
# min results are:
# Accuracy 97.51%
# Precision 52.68%
# Recall 33.52%
# F1 score 40.97%
# ROC AUC score 95.73%
#
#
# std results are:
# Accuracy 0.00%
# Precision 0.00%
# Recall 0.00%
# F1 score 0.00%
# ROC AUC score 0.00%
#
#
# The full classification report is:
#               precision    recall  f1-score   support
#
#        False       0.98      0.99      0.99     66460
#         True       0.53      0.34      0.41      1760
#
#     accuracy                           0.98     68220
#    macro avg       0.75      0.66      0.70     68220
# weighted avg       0.97      0.98      0.97     68220


# #test with pytorch:
# Weighted Precision: 0.9878409670266913
# Weighted Recall: 0.9873903859694041
# Weighted F1-score: 0.9876111436841231
# Macro Precision: 0.6740112606261313
# Macro Recall: 0.6866393071802495
# Macro F1-score: 0.6800956470807735
# Micro Precision: 0.9873903859694041
# Micro Recall: 0.9873903859694041
# Micro F1-score: 0.9873903859694041
# Recall for class 0: 0.9932786143604989
# Recall for class 1: 0.38


#test with dropouts: - better results

# Weighted Precision: 0.9878975191754963
# Weighted Recall: 0.9836139025795302
# Weighted F1-score: 0.9855436267132802
# Macro Precision: 0.638759981409608
# Macro Recall: 0.724344988043689
# Macro F1-score: 0.6709780639318811
# Micro Precision: 0.9836139025795302
# Micro Recall: 0.9836139025795302
# Micro F1-score: 0.9836139025795302
# Recall for class 0: 0.988689976087378
# Recall for class 1: 0.46



#test with dropouts and resampling minorty group: - better results

# Weighted Precision: 0.9880684520071987
# Weighted Recall: 0.9822057223324585
# Weighted F1-score: 0.9848008966714127
# Macro Precision: 0.6315819170625804
# Macro Recall: 0.740139167151382
# Macro F1-score: 0.6691988617841594
# Micro Precision: 0.9822057223324585
# Micro Recall: 0.9822057223324585
# Micro F1-score: 0.9822057223324585
# Recall for class 0: 0.9869450009694306
# Recall for class 1: 0.49333333333333335


# #test with dropouts and resampling minorty group and dialted: - better results
# Weighted Precision: 0.9878313866875356
# Weighted Recall: 0.9555783140241951
# Weighted F1-score: 0.969793044744182
# Macro Precision: 0.5617849794285998
# Macro Recall: 0.7894157564790281
# Macro F1-score: 0.5942545640159748
# Micro Precision: 0.9555783140241951
# Micro Recall: 0.9555783140241951
# Micro F1-score: 0.9555783140241951
# Recall for class 0: 0.958831512958056
# Recall for class 1: 0.62

#this is results for window = 1
# Weighted Precision: 0.9736260298152839
# Weighted Recall: 0.8276873167384117
# Weighted F1-score: 0.8914465079756182
# Macro Precision: 0.515944042303774
# Macro Recall: 0.6402926539487005
# Macro F1-score: 0.49153297232315535
# Micro Precision: 0.8276873167384117
# Micro Recall: 0.8276873167384117
# Micro F1-score: 0.8276873167384117
# Recall for class 0: 0.8340204224012178
# Recall for class 1: 0.44656488549618323

# without resampling window = 5
# Weighted Precision: 0.9713464397980148
# Weighted Recall: 0.9017106842737095
# Weighted F1-score: 0.9323700232128818
# Macro Precision: 0.5435075656458045
# Macro Recall: 0.6992725131247933
# Macro F1-score: 0.5556308909891186
# Micro Precision: 0.9017106842737095
# Micro Recall: 0.9017106842737095
# Micro F1-score: 0.9017106842737095
# Recall for class 0: 0.9099954079289759
# Recall for class 1: 0.48854961832061067

# without resampling window = 1
# Weighted Precision: 0.9551875052842871
# Weighted Recall: 0.7303799325216371
# Weighted F1-score: 0.8196768912274685
# Macro Precision: 0.5211952686964636
# Macro Recall: 0.6477648481184828
# Macro F1-score: 0.4749003665491774
# Micro Precision: 0.7303799325216371
# Micro Recall: 0.7303799325216371
# Micro F1-score: 0.7303799325216371
# Recall for class 0: 0.7355296962369654
# Recall for class 1: 0.56

# without resampling as sampling
# Weighted Precision: 0.9440531537924188
# Weighted Recall: 0.9662608185418806
# Weighted F1-score: 0.9545024835388698
# Macro Precision: 0.5148164528969482
# Macro Recall: 0.5025819857941665
# Macro F1-score: 0.49996491899476975
# Micro Precision: 0.9662608185418806
# Micro Recall: 0.9662608185418806
# Micro F1-score: 0.9662608185418806
# Recall for class 0: 0.995163971588333
# Recall for class 1: 0.01

#change the train - test
# window = 500 test on other trails
# Weighted Precision: 0.8820272614327759
# Weighted Recall: 0.8492540636829214
# Weighted F1-score: 0.8648230658817501
# Macro Precision: 0.5170274762072542
# Macro Recall: 0.5259477147963576
# Macro F1-score: 0.5180346626186998
# Micro Precision: 0.8492540636829214
# Micro Recall: 0.8492540636829214
# Micro F1-score: 0.8492540636829214
# Recall for class 0: 0.8984384312790559
# Recall for class 1: 0.15345699831365936







#approch 6, without resmapling, test on diffrent scans. window = 10

# Weighted Precision: 0.977229036852502
# Weighted Recall: 0.8772401433691757
# Weighted F1-score: 0.9233732061858959
# Macro Precision: 0.5068201935916077
# Macro Recall: 0.5571464749061269
# Macro F1-score: 0.48929698112367226
# Micro Precision: 0.8772401433691757
# Micro Recall: 0.8772401433691757
# Micro F1-score: 0.8772401433691757
# Recall for class 0: 0.8853772871616513
# Recall for class 1: 0.2289156626506024

#approch 6, without resmapling - wieghted window size, test on diffrent scans. window = 10

#
# Weighted Precision: 0.9773218041392566
# Weighted Recall: 0.9177120669056152
# Weighted F1-score: 0.9458014902939317
# Macro Precision: 0.5105932859909619
# Macro Recall: 0.5597909557479882
# Macro F1-score: 0.5059359393929573
# Micro Precision: 0.9177120669056152
# Micro Recall: 0.9177120669056152
# Micro F1-score: 0.9177120669056152
# Recall for class 0: 0.926810827158627
# Recall for class 1: 0.1927710843373494