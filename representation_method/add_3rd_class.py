from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from representation_method.main import create_method_name
from representation_method.models.autoencoder import CNNRecurrentAutoencoder
from representation_method.models.classifier import ComplexCNNClassifier, CombinedModel
from representation_method.utils.data_utils import create_dynamic_time_series_3rd_class, \
    split_train_test_for_time_series
from representation_method.utils.trainers import EnsembleTrainer, EnsembleTrainer3dClass
from utils.general_utils import seed_everything


from sklearn.manifold import TSNE

import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score
from sklearn.preprocessing import StandardScaler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colormaps

cmap = colormaps['tab10']
from representation_method.utils.data_loader import DataConfig, load_eye_tracking_data

import seaborn as sns


TRAIN  =True


def explore_agg(aggregated_features, feature_to_test):
    # Define a color mapping for the three classes.
    target_colors = {0: 'blue', 1: 'red', 2: 'green'}

    # Create subplots: one row per feature for distribution plots.
    num_features = len(feature_to_test)
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(8, num_features * 3))
    if num_features == 1:
        axes = [axes]

    # Plot distribution for each feature, for each target class.
    for ax, feature in zip(axes, feature_to_test):
        for target_val, color in target_colors.items():
            sns.histplot(aggregated_features[aggregated_features['target'] == target_val][feature],
                         color=color, kde=True, stat="density",
                         label=f'target {target_val}', ax=ax, bins=20, alpha=0.6)
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
    plt.tight_layout()
    plt.show()

    # Perform t-SNE clustering on the selected features.
    X = aggregated_features[feature_to_test].values
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(X)
    aggregated_features['tsne-2d-one'] = tsne_result[:, 0]
    aggregated_features['tsne-2d-two'] = tsne_result[:, 1]

    # Plot the t-SNE results, coloring each point by its target class.
    plt.figure(figsize=(8, 8))
    for target_value, color in target_colors.items():
        indices = aggregated_features['target'] == target_value
        plt.scatter(aggregated_features.loc[indices, 'tsne-2d-one'],
                    aggregated_features.loc[indices, 'tsne-2d-two'],
                    c=color,
                    label=f'target {target_value}',
                    alpha=0.6)
    plt.title("t-SNE Cluster Plot")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    seed = 0
    seed_everything(seed)
    use_legacy = True
    if use_legacy:
        config = DataConfig(
            data_path='data/Categorized_Fixation_Data_1_18.csv',
            approach_num=6,
            normalize=True,
            per_slice_target=True,
            participant_id=1
        )
    else:
        config = DataConfig(
            data_path='data/Formatted_Samples_ML',
            approach_num=8,
            normalize=True,
            per_slice_target=True,
            participant_id=1,
            window_size=1,
            stride=1
        )

    params = {
        'name': 'cnn with complex classifier',
        'window_size': 100,
        'classification_epochs': 200,
        'batch_size': 32,
        'ae_epochs': 150,
        'depth': 4,
        'num_filters': 32,
        'mask_probability': 0.8,
        'threshold': 0.5,
        'participant': config.participant_id,  # Added this back as it's needed for method name
        'use_gan': False,
        'early_stopping_patience': 10,
        'resample': False,
        'TRAIN': True,
        "classifier": "add dropout =0.1",
        "margin" :0.1,
        "distance_metric" :'cosine',
        "triplet_weight" :0.5,
        'lr' :0.0001,
        'latent_dim' :32,
        'seed':seed
    }


    df = load_eye_tracking_data(
        data_path=config.data_path,
        approach_num=config.approach_num,
        participant_id=config.participant_id,
        data_format="legacy"
    )

    def process_trial_group(group):
        # Group the data by slice within the trial using the image file.
        slices_info = group.groupby('CURRENT_FIX_COMPONENT_IMAGE_FILE').agg({
            'CURRENT_FIX_COMPONENT_IMAGE_NUMBER': 'first',
            'target': 'max'
        }).reset_index()

        # Sort the slices by their order within the trial.
        slices_info = slices_info.sort_values('CURRENT_FIX_COMPONENT_IMAGE_NUMBER').reset_index(drop=True)

        # For each slice that is a target (1), update the next 4 slices (if not already target) to 2.
        for i, row in slices_info.iterrows():
            if row['target'] == 1:
                for j in range(i + 1, min(i + 5, len(slices_info))):
                    if slices_info.loc[j, 'target'] != 1:
                        slices_info.loc[j, 'target'] = 2

        # Create a mapping from the slice file to its updated target value.
        slice_target_mapping = slices_info.set_index('CURRENT_FIX_COMPONENT_IMAGE_FILE')['target'].to_dict()

        # Map the updated target back to each row in the group.
        group['target'] = group['CURRENT_FIX_COMPONENT_IMAGE_FILE'].map(slice_target_mapping)
        return group
    feature_columns = [
        'Pupil_Size', 'CURRENT_FIX_DURATION', 'relative_x', 'relative_y',
        'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT', 'gaze_velocity'
    ]

    method = create_method_name(params['name'], config, params)

    method_dir = os.path.join('3d_class_results', method)
    os.makedirs(method_dir, exist_ok=True)
    df = df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'], group_keys=False).apply(process_trial_group)
    df['target'] = df['target'].replace({False: 0, True: 1})
    counts = df['target'].value_counts()
    print("Target counts:")
    print(counts)
    # explore_agg(df, feature_columns)

    print(f"window_size is {params['window_size']}")
    train_df, test_df = split_train_test_for_time_series(df, test_size=0.2, random_state=seed)
    train_df, val_df = split_train_test_for_time_series(train_df, test_size=0.2, random_state=seed)

    print("Original class distribution in test set:")
    print(test_df["target"].value_counts())
    # Create time series
    X_train, Y_train ,_= create_dynamic_time_series_3rd_class(
        train_df,feature_columns=None,participant_id=config.participant_id,load_existing=False,split_type='train',window_size=params['window_size'])
    X_test, Y_test,_ = create_dynamic_time_series_3rd_class(
        test_df,feature_columns=None,participant_id=config.participant_id,load_existing=False,split_type='test',window_size=params['window_size'])
    X_val, Y_val,_ = create_dynamic_time_series_3rd_class(
        val_df,feature_columns=None,participant_id=config.participant_id,load_existing=False,split_type='test',window_size=params['window_size'])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    input_dim = X_train.shape[-1]
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).permute(0, 2, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).permute(0, 2, 1)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, Y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

    val_loader = DataLoader(val_dataset,  batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset,  batch_size=params['batch_size'], shuffle=False)

    ensemble_save_path = os.path.join(method_dir, 'ensemble_models')

    ensemble_trainer = EnsembleTrainer3dClass(
        base_model_class=CombinedModel,
        model_params={'input_dim': input_dim,'output_classes': 3},
        n_models=10,
        device='cuda',
        save_path=ensemble_save_path

    )
    ensemble_trainer.train_ensemble(
        train_dataset=train_dataset,
        val_loader=val_loader,
        batch_size=32,
        epochs=50,
        criterion=nn.CrossEntropyLoss(),
        optimizer_class=optim.Adam,
        optimizer_params={'lr': 0.001}
    )
    accuracy, f1 = ensemble_trainer.evaluate(test_loader)
