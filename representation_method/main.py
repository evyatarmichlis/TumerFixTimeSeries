import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight

# Import our utility modules
from utils.general_utils import seed_everything
from utils.data_utils import create_time_series, split_train_test_for_time_series
from utils.losses import WeightedCrossEntropyLoss
from utils.gan_utils import generate_balanced_data_with_gan
from utils.data_loader import load_eye_tracking_data, DataConfig
from models.autoencoder import CNNRecurrentAutoencoder, initialize_weights
from models.classifier import CombinedModel
from utils.trainers import AutoencoderTrainer, CombinedModelTrainer


feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X', 'CURRENT_FIX_IA_Y',
                   'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT']


def main_with_autoencoder(df, window_size=5, method='', resample=False, classification_epochs=20, batch_size=32,
                          ae_epochs=100, depth=4, num_filters=32, lr=0.001, mask_probability=0.4,
                          early_stopping_patience=30, threshold=0.5, use_gan=False,TRAIN = False):
    """
    Main training pipeline with autoencoder and optional GAN augmentation.
    """
    # 1. SetupRemote Python 3.10.13 (sftp://evyatar613@cas602.cs.huji.ac.il:22/cs/labs/josko/evyatar613/Pycharm/TumerFixTimeSeries/tumer_venv/bin/python)
    seed_everything(0)
    interval = '30ms'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create results directory
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
        train_df, interval, window_size=window_size, resample=resample)
    X_test, Y_test, window_weight_test = create_time_series(
        test_df, interval, window_size=window_size, resample=resample)

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
        optimizer = optim.AdamW(autoencoder.parameters(), lr=0.001, weight_decay=1e-4)
        # Train autoencoder
        autoencoder_trainer = AutoencoderTrainer(
            model=autoencoder,
            criterion=nn.MSELoss(),
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20),
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
        f"_thresh_{params['threshold']}"
    )


if __name__ == '__main__':
    # Load dataset
    name = 'GRU-AE'
    config = DataConfig(
        data_path='data/Formatted_Samples_ML',
        approach_num=6,
        normalize=True,
        per_slice_target=True,
        participant_id=1
    )

    # Define all parameters
    params = {
        'window_size': 50,
        'classification_epochs': 1,
        'batch_size': 32,
        'ae_epochs': 1,
        'depth': 4,
        'num_filters': 32,
        'lr': 0.001,
        'mask_probability': 0.4,
        'threshold': 0.9,
        'participant': 1,
        'use_gan': False,
        'early_stopping_patience': 30,
        'resample': False
    }

    # Load and filter data
    df = load_eye_tracking_data(data_path=config.data_path,approach_num=config.approach_num,participant_id=config.participant_id)

    # Create method name from parameters
    method_name = create_method_name(name,config, params)

    # Run training
    main_with_autoencoder(
        df=df,
        window_size=params['window_size'],
        method=method_name,
        classification_epochs=params['classification_epochs'],
        batch_size=params['batch_size'],
        ae_epochs=params['ae_epochs'],
        depth=params['depth'],
        num_filters=params['num_filters'],
        lr=params['lr'],
        mask_probability=params['mask_probability'],
        threshold=params['threshold'],
        use_gan=params['use_gan'],
        early_stopping_patience=params['early_stopping_patience'],
        resample=params['resample'],TRAIN=True
    )