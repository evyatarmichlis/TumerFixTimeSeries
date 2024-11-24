import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight

from data_process import IdentSubRec
from representation_method.utils.visualization import analyze_embeddings_from_loader
# Import our utility modules
from utils.general_utils import seed_everything
from utils.data_utils import create_time_series, split_train_test_for_time_series, DataSplitter
from utils.losses import WeightedCrossEntropyLoss, ContrastiveAutoencoderLoss
from utils.gan_utils import generate_balanced_data_with_gan
from utils.data_loader import load_eye_tracking_data, DataConfig, create_data_loader
from models.autoencoder import CNNRecurrentAutoencoder, initialize_weights
from models.classifier import CombinedModel
from utils.trainers import AutoencoderTrainer, CombinedModelTrainer, ContrastiveAutoencoderTrainer

feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X', 'CURRENT_FIX_IA_Y',
                   'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT']


def main_with_autoencoder(df, window_size=5, method='', resample=False, classification_epochs=20, batch_size=32,
                          ae_epochs=100, depth=4, num_filters=32, lr=0.001, mask_probability=0.4,
                          early_stopping_patience=30, threshold=0.5, use_gan=False,TRAIN = False):
    """
    Main training pipeline with autoencoder and optional GAN augmentation.
    """
    seed_everything(0)
    interval = '30ms'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create results_old directory
    method_dir = os.path.join('results', method)
    os.makedirs(method_dir, exist_ok=True)

    # Save hyperparameters
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

    with open(os.path.join(method_dir, 'hyperparameters.txt'), 'w') as f:
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")

    # 2. Data Preparation
    # Split train/test
    train_df, test_df = split_train_test_for_time_series(df, test_size=0.2, random_state=0)
    print("Original class distribution in test set:")
    print(test_df["target"].value_counts())

    # Create time series
    X_train_full, Y_train_full, window_weight_train_full = create_time_series(
        train_df,config.participant_id, interval, window_size=window_size, resample=resample,load_existing=True)
    X_test, Y_test, window_weight_test = create_time_series(
        test_df,config.participant_id, interval, window_size=window_size, resample=resample,load_existing=True)
    print("Original class distribution in test set:")
    print(pd.Series(Y_test).value_counts())
    # Split training/validation
    X_train, X_val, Y_train, Y_val, window_weight_train, window_weight_val = train_test_split(
        X_train_full, Y_train_full, window_weight_train_full,
        test_size=0.2, random_state=42, stratify=Y_train_full)

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # 3. Autoencoder Training
    autoencoder = CNNRecurrentAutoencoder(
        in_channels=len(feature_columns),
        num_filters=num_filters,
        depth=depth,
        hidden_size=128,
        num_layers=1,
        rnn_type='GRU',
        input_length=X_train.shape[1]
    ).to(device)

    autoencoder.apply(initialize_weights)
    best_autoencoder_path = os.path.join(method_dir, 'best_autoencoder.pth')

    if TRAIN:
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).permute(0, 2, 1)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
        Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, Y_val_tensor)

        train_loader_ae = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader_ae = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        optimizer = optim.AdamW(autoencoder.parameters(), lr=0.001, weight_decay=0.01)
        # Train autoencoder
        autoencoder_trainer = AutoencoderTrainer(
            model=autoencoder,
            criterion=nn.MSELoss(),
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5),
            device=device,
            save_path=method_dir,
            early_stopping_patience=20,
            mask_probability = mask_probability
        )

        autoencoder_trainer.train(
            train_loader_ae,
            val_loader_ae,
            epochs=ae_epochs,
        )

    best_autoencoder_path = '/cs/usr/evyatar613/Desktop/josko_lab/Pycharm/TumerFixTimeSeries/representation_method/results/Evyatar_approach_6_window_50_depth_4_lr_0.001_ae_epochs_100_class_epochs_20_mask_0.4_filters_32_batch_32_participant_1_thresh_0.9,use_gan_True/best_autoencoder_model.pth'
    autoencoder.load_state_dict(torch.load(best_autoencoder_path))
    # 4. GAN Data Generation (if enabled)
    if use_gan:
        print("\nGenerating synthetic data using TimeGAN...")
        X_train_balanced, Y_train_balanced, window_weight_balanced = generate_balanced_data_with_gan(
            X_train_scaled, Y_train, window_weight_train, method_dir, device)
        print("Finished generating synthetic data")
    else:
        X_train_balanced = X_train_scaled
        Y_train_balanced = Y_train
        window_weight_balanced = window_weight_train

    # 5. Prepare Data for Combined Model
    X_train_tensor = torch.tensor(X_train_balanced, dtype=torch.float32).permute(0, 2, 1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).permute(0, 2, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).permute(0, 2, 1)

    Y_train_tensor = torch.tensor(Y_train_balanced, dtype=torch.long)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, Y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor)

    # Create data loaders
    if use_gan:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        # Use weighted sampling for imbalanced data
        classes = np.unique(Y_train_balanced)
        class_weights = compute_class_weight('balanced', classes=classes, y=Y_train_balanced)
        samples_weight = np.array([class_weights[t] for t in Y_train_balanced])
        sampler = WeightedRandomSampler(torch.from_numpy(samples_weight).float(), len(samples_weight))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 6. Train Combined Model
    model = CombinedModel(autoencoder, num_classes=2).to(device)

    # Setup optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': lr * 0.1},
        {'params': model.classifier.parameters(), 'lr': lr}
    ], weight_decay=1e-4)

    # Create trainer
    combined_trainer = CombinedModelTrainer(

        model=model,
        criterion=WeightedCrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.5, patience=10),
        device=device,
        save_path=method_dir,
        early_stopping_patience=5,

    )

    # Train model
    combined_trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=classification_epochs,
        window_weights_train=window_weight_balanced,
        window_weights_val=window_weight_val
    )

    # 7. Evaluate Model
    combined_trainer.evaluate(test_loader, threshold=threshold)


def create_method_name(name,config, params):
    """Create method name from parameters."""
    return (
        f"{name}"
        f"_approach_{config.approach_num}"
        f"_window_{params['window_size']}"
        f"_depth_{params['depth']}"
        f"_lr_{params['lr']}"
        f"_ae_epochs_{params['ae_epochs']}"
        f"_class_epochs_{params['classification_epochs']}"
        f"_mask_{params['mask_probability']}"
        f"_filters_{params['num_filters']}"
        f"_batch_{params['batch_size']}"
        f"_participant_{params['participant']}"
        f"_thresh_{params['threshold']},"
        f"use_gan_{params['use_gan']}"
    )



def prepare_dataloaders(X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_size, use_gan=False):
    """Create all necessary dataloaders with proper sampling."""
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1),
        torch.tensor(Y_train, dtype=torch.long)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1),
        torch.tensor(Y_val, dtype=torch.long)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1),
        torch.tensor(Y_test, dtype=torch.long)
    )

    if not use_gan:
        classes = np.unique(Y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=Y_train)
        samples_weight = np.array([class_weights[t] for t in Y_train])
        sampler = WeightedRandomSampler(torch.from_numpy(samples_weight).float(), len(samples_weight))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def setup_autoencoder(input_shape, params, device):
    """Create and initialize autoencoder."""
    autoencoder = CNNRecurrentAutoencoder(
        in_channels=input_shape[-1],
        num_filters=params['num_filters'],
        depth=params['depth'],
        hidden_size=128,
        num_layers=1,
        rnn_type='GRU',
        input_length=input_shape[1]
    ).to(device)

    autoencoder.apply(initialize_weights)
    return autoencoder


def train_autoencoder_model(autoencoder, train_loader, val_loader, params, device, save_path):
    """Setup and train autoencoder."""
    optimizer = optim.AdamW(autoencoder.parameters(), lr=params['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    # trainer = AutoencoderTrainer(
    #     model=autoencoder,
    #     criterion=nn.MSELoss(),
    #     optimizer=optimizer,
    #     scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20),
    #     device=device,
    #     save_path=save_path,
    #     early_stopping_patience=10,
    #     mask_probability=params['mask_probability']
    # )

    # trainer.train(train_loader,val_loader,epochs=params['ae_epochs'])
    # Calculate weights for balanced sampling
    labels = train_loader.dataset.tensors[1].numpy()  # Get labels from dataset
    class_counts = np.bincount(labels)
    weights = 1. / class_counts[labels]
    weights = torch.FloatTensor(weights)

    # Create balanced sampler
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    # Create new train loader with balanced sampling
    train_loader_balanced = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=sampler,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory
    )

    # Your existing code
    criterion = nn.MSELoss()
    loss_function = ContrastiveAutoencoderLoss(
        criterion=criterion,
        lambda_contrast=0.05,
        temperature=0.5
    )

    # Create trainer with chosen loss
    contrastive_trainer = ContrastiveAutoencoderTrainer(
        model=autoencoder,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
        scheduler=scheduler,
        mask_probability=0.1,
        save_path=save_path,
        early_stopping_patience=5
    )
    #
    # # Train with balanced loader
    contrastive_trainer.train(train_loader, val_loader, epochs=params['ae_epochs'])


def train_combined_model(autoencoder, train_loader, val_loader, test_loader, params, device, save_path):
    """Setup and train combined model."""
    model = CombinedModel(autoencoder, num_classes=2).to(device)
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': params['lr'] * 0.1},
        {'params': model.classifier.parameters(), 'lr': params['lr']}
    ], weight_decay=1e-4)
    classifier_dir = os.path.join(save_path, params['classifier'])
    os.makedirs(classifier_dir, exist_ok=True)
    trainer = CombinedModelTrainer(
        model=model,
        criterion=CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3,verbose=True),
        device=device,
        save_path=classifier_dir,
        early_stopping_patience=5
    )

    trainer.train(train_loader, val_loader, epochs=params['classification_epochs'])
    trainer.evaluate(test_loader, threshold=params['threshold'])


def main(data_config, params, use_legacy=False):
    """Main training pipeline."""
    seed_everything(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    method_name = create_method_name(params['name'], data_config, params)
    method_dir = os.path.join('results', method_name)
    os.makedirs(method_dir, exist_ok=True)

    if use_legacy:
        df = load_eye_tracking_data(data_path=data_config.data_path,
                                    approach_num=data_config.approach_num,
                                    participant_id=data_config.participant_id,
                                    data_format="legacy")
        return main_with_autoencoder(df=df, window_size=params['window_size'],
                                     method=method_name,
                                     **{k: v for k, v in params.items() if k != 'name'})

    # Load and split data
    loader = create_data_loader('time_series', data_config)
    window_data, labels, meta_data = loader.load_data(data_type='windowed')
    splitter = DataSplitter(window_data, labels, meta_data, random_state=42)
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = splitter.split_by_trials()

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Create dataloaders
    train_loader, val_loader, test_loader = prepare_dataloaders(
        X_train_scaled, Y_train, X_val_scaled, Y_val, X_test_scaled, Y_test,
        params['batch_size'], params['use_gan']
    )

    # Setup and train autoencoder
    autoencoder = setup_autoencoder(X_train.shape, params, device)
    if params['TRAIN']:
        train_autoencoder_model(autoencoder, train_loader, val_loader, params, device, method_dir)
        analyze_embeddings_from_loader(
            encoder=autoencoder.encoder,
            loader=test_loader,
            device=device,
            method_dir=method_dir,
            loader_name="Test"
        )

    else:

        checkpoint = torch.load(os.path.join(method_dir, 'best_autoencoder_checkpoint.pth'))
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        # path = r'C:\Users\Evyatar\PycharmProjects\TumorFixTimeSeriesUpdate\representation_method\results\new_data normal loss add regulaztion_approach_6_window_1000_depth_5_lr_0.0001_ae_epochs_50_class_epochs_20_mask_0.4_filters_4_batch_64_participant_1_thresh_0.9,use_gan_False\best_autoencoder_model.pth'
        # autoencoder.load_state_dict(torch.load(os.path.join(method_dir, 'best_autoencoder_checkpoint.pth')))

        analyze_embeddings_from_loader(
            encoder=autoencoder.encoder,
            loader=test_loader,
            device=device,
            method_dir=method_dir,
            loader_name="Test"
        )


    # Handle GAN augmentation if enabled
    if params['use_gan']:
        X_train_balanced, Y_train_balanced = generate_balanced_data_with_gan(
            X_train_scaled, Y_train, method_dir, device)
        train_loader, _, _ = prepare_dataloaders(
            X_train_balanced, Y_train_balanced, X_val_scaled, Y_val, X_test_scaled, Y_test,
            params['batch_size'], True
        )
    else:
        X_train_balanced = X_train_scaled
        Y_train_balanced = Y_train
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).permute(0, 2, 1)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
        Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, Y_val_tensor)
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, Y_val_tensor)
        classes = np.unique(Y_train_balanced)
        class_weights = compute_class_weight('balanced', classes=classes, y=Y_train_balanced)
        samples_weight = np.array([class_weights[t] for t in Y_train_balanced])
        sampler = WeightedRandomSampler(torch.from_numpy(samples_weight).float(), len(samples_weight))
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], sampler=sampler)

    train_combined_model(autoencoder, train_loader, val_loader, test_loader, params, device, method_dir)





if __name__ == '__main__':
    # Load either legacy or new data based on flag
    use_legacy = False

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
            approach_num=6,
            normalize=True,
            per_slice_target=True,
            participant_id=1,
            window_size=1000,
            stride=1
        )

    params = {
        'name': 'new_data cont loss add more lambda to cont',
        'window_size': 1000,
        'classification_epochs': 20,
        'batch_size': 64,
        'ae_epochs': 50,
        'depth': 5,
        'num_filters': 4,
        'lr': 0.0001,
        'mask_probability': 0.4,
        'threshold': 0.9,
        'participant': 1,
        'use_gan': False,
        'early_stopping_patience': 30,
        'resample': False,
        'TRAIN': True,
        "classifier":"add dropout =0.1"
    }

    main(config, params, use_legacy)

