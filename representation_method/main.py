import json
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pyparsing import alphas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sympy.benchmarks.bench_meijerint import alpha
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

from data_process import IdentSubRec
from representation_method.utils.statistical_classifer import analyze_dynamic_windows
from representation_method.utils.trainers import FocalLoss, EnsembleTrainer
from representation_method.utils.visualization import analyze_embeddings_from_loader, \
    analyze_vae_embeddings_from_loader, analyze_vae_embeddings_with_umap, analyze_encoder_embeddings_with_umap_2
# Import our utility modules
from utils.general_utils import seed_everything
from utils.data_utils import create_time_series, split_train_test_for_time_series, DataSplitter, \
    create_dynamic_time_series, find_max_consecutive_hits
from utils.losses import WeightedCrossEntropyLoss, ContrastiveAutoencoderLoss, ImbalancedTripletContrastiveLoss, \
    CombinedTemporalLoss
from utils.gan_utils import generate_balanced_data_with_gan
from utils.data_loader import load_eye_tracking_data, DataConfig, create_data_loader
from models.autoencoder import CNNRecurrentAutoencoder, initialize_weights, TimeSeriesVAE, ImprovedTimeSeriesVAE, \
    CombinedVAEClassifier
from models.classifier import CombinedModel, TVAEClassifier, EnhancedClassifier, EnhancedCombinedModel, \
    SimpleCombinedModel
from utils.trainers import AutoencoderTrainer, CombinedModelTrainer, ContrastiveAutoencoderTrainer, VAETrainer, \
    VAEClassifierTrainer, ImprovedVAEClassifierTrainer, TripletAutoencoderTrainer
from latent_space_classifer import  LatentSpaceClassifier
feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X', 'CURRENT_FIX_IA_Y',
                   'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT']

# feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION']
def main_with_autoencoder(df, window_size=5, method='', resample=False, classification_epochs=20, batch_size=32,
                          ae_epochs=100, depth=4, num_filters=32, lr=0.001, mask_probability=0.4,
                          early_stopping_patience=30, threshold=0.5, use_gan=False, TRAIN = False, use_vae=True,
                          vae_params=None, assemble=False):
    """
    Main training pipeline with autoencoder and optional GAN augmentation.
    """
    seed_everything(0)
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
        'use_gan': use_gan,
        'use_vae': use_vae
    }
    if use_vae and vae_params:
        hyperparameters.update(vae_params)

    with open(os.path.join(method_dir, 'hyperparameters.txt'), 'w') as f:
        for key, value in hyperparameters.items():
            f.write(f"{key}: {value}\n")

    # 2. Data Preparation
    # Split train/test
    # max_consecutive = find_max_consecutive_hits(df)
    # window_size = max_consecutive + 4
    # Add buffer
    print(f"window_size is {window_size}")
    train_df, test_df = split_train_test_for_time_series(df, test_size=0.2, random_state=0)
    train_df, val_df = split_train_test_for_time_series(train_df, test_size=0.2, random_state=0)

    print("Original class distribution in test set:")
    print(test_df["target"].value_counts())



    # Create time series
    X_train, Y_train = create_dynamic_time_series(
        train_df,feature_columns=None,participant_id=config.participant_id,load_existing=False,split_type='train',window_size=window_size)
    X_test, Y_test = create_dynamic_time_series(
        test_df,feature_columns=None,participant_id=config.participant_id,load_existing=False,split_type='test',window_size=window_size)
    X_val, Y_val = create_dynamic_time_series(
        val_df,feature_columns=None,participant_id=config.participant_id,load_existing=False,split_type='test',window_size=window_size)
    print("Original class distribution in test set:")
    print(pd.Series(Y_test).value_counts())
    # results, best_clf = analyze_dynamic_windows(X_train, Y_train, X_test, Y_test, X_val, Y_val)
    # return results
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    if use_vae:
        autoencoder = TimeSeriesVAE(
            input_dim=len(feature_columns)+2,
            hidden_dim=64,
            latent_dim=vae_params.get('latent_dim', 32)
        ).to(device)

        # autoencoder = ImprovedTimeSeriesVAE(len(feature_columns)+2,latent_dim=vae_params.get('latent_dim', 32)).to(device)
    else:
        autoencoder =CNNRecurrentAutoencoder(
            in_channels=len(feature_columns)+2,  # Number of input features
            num_filters=32,  # Match the old model's filter count
            depth=2,  # - original 4
            hidden_size=128,  # Match the old hidden size
            num_layers=1,
            rnn_type='GRU',
            input_length=window_size
        ).to(device)



    autoencoder.apply(initialize_weights)
    best_autoencoder_path = os.path.join(method_dir, 'best_autoencoder.pth')

    if TRAIN:
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).permute(0, 2, 1)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).permute(0, 2, 1)

        # Force labels to be 1D
        Y_train_1d = Y_train.flatten() if isinstance(Y_train, np.ndarray) else np.array(Y_train).flatten()
        Y_val_1d = Y_val.flatten() if isinstance(Y_val, np.ndarray) else np.array(Y_val).flatten()
        Y_test_1d = Y_test.flatten() if isinstance(Y_test, np.ndarray) else np.array(Y_test).flatten()

        print("Shape after flattening:", Y_train_1d.shape)  # Debug print

        # Create tensors from 1D arrays
        Y_train_tensor = torch.tensor(Y_train_1d, dtype=torch.long)
        Y_val_tensor = torch.tensor(Y_val_1d, dtype=torch.long)
        Y_test_tensor = torch.tensor(Y_test_1d, dtype=torch.long)

        # Create datasets
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, Y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, Y_val_tensor)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor)

        # Calculate weights
        unique_labels = np.unique(Y_train_1d)
        class_counts = np.bincount(Y_train_1d.astype(int))
        weights = np.zeros(len(Y_train_1d), dtype=np.float32)

        for label in unique_labels:
            mask = (Y_train_1d == label)
            weights[mask] = 1.0 / class_counts[int(label)]

        # Convert to tensor and create sampler
        weights = torch.FloatTensor(weights)

        # Convert to tensor and create sampler
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Setup optimizer and scheduler
        optimizer = optim.AdamW(autoencoder.parameters(), lr=params.get('lr', 0.001), weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        # Train autoencoder
        if use_vae:
            trainer = VAETrainer(
                model=autoencoder,
                criterion=nn.MSELoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                l2_alpha=params.get('alpha', 0.1),
                beta=params.get('beta', 0.1),
                margin=params.get('margin', 0.1),
                triplet_weight=params.get('triplet_weight', 30),
                distance_metric=params.get('distance_metric', 'cosine'),
                loss_type=params.get('loss_type', 'normal'),
                save_path=method_dir,
                early_stopping_patience=early_stopping_patience
            )
        else:
            trainer = TripletAutoencoderTrainer(
                model=autoencoder,
                criterion=nn.MSELoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                margin=params.get('margin', 10),
                triplet_weight=params.get('triplet_weight', 30),
                distance_metric=params.get('distance_metric', 'cosine'),
                mask_probability=params.get('mask_probability', 0.1),
                save_path=method_dir,
                early_stopping_patience=early_stopping_patience

                )
            trainer = AutoencoderTrainer(
                model=autoencoder,
                criterion=nn.MSELoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                mask_probability=params.get('mask_probability', 0.4),
                save_path=method_dir,
                early_stopping_patience=early_stopping_patience

            )



        trainer.train(train_loader, val_loader, epochs=ae_epochs)
        # best_autoencoder_path = '/cs/usr/evyatar613/Desktop/josko_lab/Pycharm/TumerFixTimeSeries/representation_method/results/VAET old method improved_approach_6_window_50_depth_5_lr_0.001_ae_epochs_30_class_epochs_20_mask_0.4_filters_4_batch_32_participant_1_thresh_0.95,use_gan_False/best_model_model.pth'
        # # best_autoencoder_path = '/cs/usr/evyatar613/Desktop/josko_lab/Pycharm/TumerFixTimeSeries/representation_method/results/AE on old data  test_approach_6_window_50_depth_5_lr_0.001_ae_epochs_30_class_epochs_30_mask_0.4_filters_4_batch_256_participant_1_thresh_0.95,use_gan_False/best_autoencoder_checkpoint.pth'
        # # best_autoencoder_path = '/cs/usr/evyatar613/Desktop/josko_lab/Pycharm/TumerFixTimeSeries/representation_method/results/Evyatar_approach_6_window_50_depth_4_lr_0.001_ae_epochs_100_class_epochs_20_mask_0.4_filters_32_batch_32_participant_1_thresh_0.9,use_gan_True/best_autoencoder_model.pth'
        # autoencoder.load_state_dict(torch.load(best_autoencoder_path))
        # results = analyze_vae_embeddings_with_umap(
        #     encoder_model=autoencoder,
        #     loader=test_loader,
        #     device=device,
        #     method_dir=method_dir,
        #     loader_name='test',
        #     n_iterations=5,
        #     sample_ratio=0.2
        # )
        # return results
    else:
        root_dir = Path(__file__).resolve().parent  # Go up 3 levels
        best_autoencoder_path = os.path.join(root_dir, "results",'dynamic windowing with diffs assemble - focal loss 0.75_approach_6_participant_1use_gan_False',
                                             'best_autoencoder_model.pth')


        autoencoder.load_state_dict(torch.load(best_autoencoder_path))

    # 4. GAN Data Generation (if enabled)

    if use_gan:
        print("\nGenerating synthetic data using TimeGAN...")
        X_train_balanced, Y_train_balanced = generate_balanced_data_with_gan(
            X_train_scaled, Y_train, method_dir, device)
        print("Finished generating synthetic data")
    else:
        X_train_balanced = X_train_scaled
        Y_train_balanced = Y_train

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
        Y_train_1d = Y_train.flatten() if isinstance(Y_train, np.ndarray) else np.array(Y_train).flatten()
        unique_labels = np.unique(Y_train_1d)
        class_counts = np.bincount(Y_train_1d.astype(int))
        weights = np.zeros(len(Y_train_1d), dtype=np.float32)
        for label in unique_labels:
            mask = (Y_train_1d == label)
            weights[mask] = 1.0 / class_counts[int(label)]

        # Convert to tensor and create sampler
        weights = torch.FloatTensor(weights)

        # Convert to tensor and create sampler
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # model = TVAEClassifier(autoencoder, num_classes=2).to(device)

    if assemble:
        # Initialize ensemble trainer
        ensemble_trainer = EnsembleTrainer(
            base_model_class=EnhancedCombinedModel,  # Your model class
            model_params={'model': autoencoder, 'num_classes': 2},
            n_models=10,  # Number of models in ensemble
            device='cuda',
            save_path='ensemble_models'
        )

        # Train ensemble

        ensemble_trainer.train_ensemble(
            train_dataset=train_dataset,
            val_loader=val_loader,
            batch_size=32,
            epochs=50,
            criterion=nn.CrossEntropyLoss(),
            optimizer_class=optim.Adam,
            optimizer_params={'lr': 0.001},majority_weight=weights[0]
        )
        accuracy, f1= ensemble_trainer.evaluate(test_loader)
        return accuracy,f1
    model = EnhancedCombinedModel(autoencoder, num_classes=2).to(device)
    # TVAEClassifier
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': lr * 0.1},
        {'params': model.classifier.parameters(), 'lr': lr}
    ], weight_decay=1e-4)

    # Create trainer
    try:
        classifier, test_metrics = train_latent_classifier(vae_model=autoencoder, train_loader=train_loader,
                                                           val_loader=val_loader, test_loader=test_loader, device=device)
        print(test_metrics)
    except Exception:
        print("error in latent classifier")

    criterion = CrossEntropyLoss()
    combined_trainer = CombinedModelTrainer(
        model=model,
        criterion=criterion,
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
        epochs=classification_epochs
    )

    root_dir = Path(__file__).resolve().parent  # Go up 3 levels

    # model_path   = os.path.join(root_dir, "results",'dynamic windowing with diffs normal AE_approach_8_participant_9use_gan_True',
    #                                          'best_classifier_model.pth')
    # autoencoder.load_state_dict(torch.load(best_autoencoder_path))

    report, cm, threshold = combined_trainer.evaluate(test_loader, threshold=threshold)
    print(report)
    print(cm)


def create_method_name(name,config, params):
    """Create method name from parameters."""
    return (
        f"{name}"
        f"_approach_{config.approach_num}"
        f"_participant_{params['participant']}"
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
    # autoencoder = CNNRecurrentAutoencoder(
    #     in_channels=input_shape[-1],
    #     num_filters=params['num_filters'],
    #     depth=params['depth'],
    #     hidden_size=128,
    #     num_layers=1,
    #     rnn_type='GRU',
    #     input_length=input_shape[1]
    # ).to(device)
    autoencoder = TimeSeriesVAE(input_dim=input_shape[-1], hidden_dim=64, latent_dim=params['latent_dim']).to(device)
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
    #
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

    trainer = VAETrainer(
        model=autoencoder,
        criterion = nn.MSELoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        l2_alpha = params['alpha'],
        beta=params['beta'],
        margin = params['margin'],
        triplet_weight = params['triplet_weight'],
        distance_metric = params['distance_metric'],
        loss_type = params['loss_type'],
        # Adjust this to control KL divergence impact
        save_path=save_path,
        early_stopping_patience=5
    )
    #
    trainer.train(train_loader_balanced, val_loader, epochs=params['ae_epochs'])
    # criterion = nn.MSELoss()
    # loss_fn = ImbalancedTripletContrastiveLoss(
    #     criterion=criterion,
    #     lambda_contrast=5.0,
    #     lambda_triplet=2.0,  # Adjust based on your needs
    #     temperature=0.1,
    #     margin=1.0  # Adjust based on your embedding space
    # ).to(device)
    # # Your existing code
    # criterion = nn.MSELoss()
    # # loss_fn = ContrastiveAutoencoderLoss(
    # #     criterion=criterion
    # # ).to(device)
    #
    # # Create trainer with chosen loss
    # contrastive_trainer = ContrastiveAutoencoderTrainer(
    #     model=autoencoder,
    #     optimizer=optimizer,
    #     loss_function=loss_fn,
    #     device=device,
    #     scheduler=scheduler,
    #     mask_probability=0.1,
    #     save_path=save_path,
    #     early_stopping_patience=5
    # )
    # #
    # # # Train with balanced loader
    # contrastive_trainer.train(train_loader_balanced, val_loader, epochs=params['ae_epochs'])


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


def train_vae_classifier(autoencoder, train_loader, val_loader, test_loader, params, device, method_dir):
    """
    Train VAE classifier using the new trainer
    """
    vae_classifier = TVAEClassifier(
        vae_model=autoencoder,
        hidden_dim=128,  # You can adjust this
        num_classes=2,
        dropout=0.3
    ).to(device)

    trainer = VAEClassifierTrainer(
        model=vae_classifier,
        device=device,
        focal_loss_gamma=2.0  # You can adjust this
    )

    _, labels = next(iter(train_loader))
    class_counts = np.bincount(labels.numpy())
    class_weights = len(labels) / (len(np.unique(labels)) * class_counts)

    # Create paths for saving results
    model_save_path = os.path.join(method_dir, 'best_vae_classifier.pth')
    curves_save_path = os.path.join(method_dir, 'vae_training_curves.png')
    results_save_path = os.path.join(method_dir, 'vae_classification_results.txt')

    # Train the model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=params['classification_epochs'],
        learning_rate=params['lr'],
        class_weights=class_weights,
        early_stopping_patience=10,
        save_path=model_save_path
    )

    trainer.plot_training_history(history, save_path=curves_save_path)

    results = trainer.evaluate(test_loader)

    with open(results_save_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(results['classification_report'])
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(results['confusion_matrix']))
    return vae_classifier, results


def train_latent_classifier(vae_model, train_loader, val_loader, test_loader, device):
    """Train and evaluate the latent space classifier"""

    # Initialize classifier
    classifier = LatentSpaceClassifier(
        vae_model=vae_model,
        n_neighbors=5,  # Increased for more robust estimation
        contamination=0.03,  # Adjust based on expected anomaly ratio
        device=device
    )

    # Train
    print("Training latent space classifier...")
    classifier.fit(train_loader, val_loader)

    # Find optimal threshold
    # best_threshold = classifier.optimize_threshold(
    #     val_loader,
    #     min_recall=0.3  # Minimum recall we want to achieve
    # )
    # print(f"\nOptimal threshold: {best_threshold:.3f}")
    best_threshold  =0.5
    # Evaluate on test set
    test_metrics = classifier.evaluate(test_loader, threshold=best_threshold)
    print("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    return classifier, test_metrics



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
        embeddings, median_embeddings_2d, median_labels, metrics = analyze_vae_embeddings_with_umap(
            vae_model=autoencoder,
            loader=test_loader,
            device=device,
            method_dir=method_dir,
            n_iterations=10,
            sample_ratio=0.2,
            n_neighbors=15,
            min_dist=0.1
        )
        return metrics['avg_davies'],metrics['avg_silhouette']


    else:
        path = '/cs/usr/evyatar613/Desktop/josko_lab/Pycharm/TumerFixTimeSeries/representation_method/results/All_VAE_m0.1_tw10_b0.1_a0.1_dist_L2_loss_normal_dim_32_lr0.001_approach_6_window_1000_depth_5_lr_0.001_ae_epochs_70_class_epochs_30_mask_0.4_filters_4_batch_512_participant_1_thresh_0.9,use_gan_False/best_model_model.pth'
        # path = '/cs/usr/evyatar613/Desktop/josko_lab/Pycharm/TumerFixTimeSeries/representation_method/results/VAE_m0.4_tw10_b0.1_a0.05_cosine_normal_approach_6_window_1000_depth_5_lr_0.0001_ae_epochs_100_class_epochs_20_mask_0.4_filters_4_batch_256_participant_1_thresh_0.9,use_gan_False/best_model_model.pth'
        checkpoint = torch.load(path)
        autoencoder.load_state_dict(checkpoint)
        classifier, test_metrics = train_latent_classifier(vae_model=autoencoder, train_loader=train_loader,val_loader=val_loader,test_loader=test_loader,device=device)


if __name__ == '__main__':


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
        'name': 'dynamic windowing with diffs assemble - keep only target samples',
        'window_size': 200,
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
        "margin":0.1,
        "distance_metric":'cosine',
        "triplet_weight":0.5,
        'lr':0.0001,
        'latent_dim':32
    }



    # Load legacy data
    df = load_eye_tracking_data(
        data_path=config.data_path,
        approach_num=config.approach_num,
        participant_id=config.participant_id,
        data_format="legacy"
    )

    # Combine params for method name
    combined_params = params  # This merges both dictionaries

    method_name = create_method_name(combined_params['name'], config, combined_params)

    main_with_autoencoder(
        df=df,
        window_size=params['window_size'],
        method=method_name,  # generated from create_method_name
        resample=params['resample'],
        classification_epochs=params['classification_epochs'],  # from params['classification_epochs']
        batch_size=params['batch_size'],  # from params['batch_size']
        ae_epochs=params['ae_epochs'],  # from params['ae_epochs']
        depth=params['depth'],  # from params['depth']
        num_filters=params['num_filters'],  # from params['num_filters']
        lr=0.001,  # from vae_params['lr']
        mask_probability=params['mask_probability'],  # from params['mask_probability']
        early_stopping_patience=params['early_stopping_patience'],  # from params['early_stopping_patience']
        threshold=params['threshold'],  # from params['threshold']
        use_gan=params['use_gan'],  # from params['use_gan']
        TRAIN=params['TRAIN'],  # from params['TRAIN']
        use_vae=False,  # since we want to use VAE
        vae_params=params , # the VAE-specific parameters dictionary
        assemble=True,
    )
    # main(config, params, use_legacy)
    #
    # import itertools
    # from copy import deepcopy
    #
    # best_results = {
    #     'silhouette': {
    #         'score': float('-inf'),
    #         'params': None
    #     },
    #     'davies': {
    #         'score': float('inf'),  # Lower is better for Davies-Bouldin
    #         'params': None
    #     }
    # }
    # all_results = []
    # # Grid parameters
    # margins = [0.1,0.3,0.5]
    # triplet_weights = [10,30,50]
    # betas = [0.1]
    # alphas = [0.05,0.1]
    # distances = ["cosine","L2"]
    # loss_types = ["normal","dual"]
    # latent_dims = [2,8,16,32,64]
    # lrs = [0.01,0.001,0.0001]
    # # Base configurations
    # use_legacy = False
    #
    # if use_legacy:
    #     base_config = DataConfig(
    #         data_path='data/Categorized_Fixation_Data_1_18.csv',
    #         approach_num=6,
    #         normalize=True,
    #         per_slice_target=True,
    #         participant_id=1
    #     )
    # else:
    #     base_config = DataConfig(
    #         data_path='data/Formatted_Samples_ML',
    #         approach_num=6,
    #         normalize=True,
    #         per_slice_target=True,
    #         participant_id=1,
    #         window_size=1000,
    #         stride=1
    #     )

    # base_params = {
    #     'name': 'VAE triple loss grid search',
    #     'window_size': 1000,
    #     'classification_epochs': 30,
    #     'batch_size': 512,
    #     'ae_epochs': 70,
    #     'depth': 5,
    #     'num_filters': 4,
    #     'mask_probability': 0.4,
    #     'threshold': 0.9,
    #     'participant': 1,
    #     'use_gan': False,
    #     'early_stopping_patience': 5,
    #     'resample': False,
    #     'TRAIN': True,
    #     'classifier': 'add dropout =0.1'
    # }
    #
    # # Run experiments
    # total_experiments = len(margins) * len(triplet_weights) * len(betas) * len(distances) * len(loss_types) *len(alphas) * len(latent_dims)
    # current_experiment = 0
    #
    # for margin, triplet_weight, beta, distance,alpha_l2, loss_type,latent_dim,lr in itertools.product(
    #         margins, triplet_weights, betas, distances,alphas, loss_types,latent_dims,lrs):
    #
    #     current_experiment += 1
    #     print(f"\nExperiment {current_experiment}/{total_experiments}")
    #     print(f"Margin: {margin}, Triplet Weight: {triplet_weight}, Beta: {beta}, Alpha {alpha_l2}")
    #     print(f"Distance: {distance}, Loss Type: {loss_type}")
    #     print(f"Latent Dimensions: {latent_dim}")
    #     print(f"LR: {lr}")
    #     # Create copies of base configurations
    #     config = deepcopy(base_config)
    #     params = deepcopy(base_params)
    #     current_params = {
    #         'margin': margin,
    #         'triplet_weight': triplet_weight,
    #         'beta': beta,
    #         'distance_metric': distance,
    #         'loss_type': loss_type,
    #         'alpha': alpha_l2,
    #         'latent_dim': latent_dim,
    #         'lr': lr
    #     }
    #     params.update(current_params)
    #     params['name'] = f'All_VAE_m{margin}_tw{triplet_weight}_b{beta}_a{alpha_l2}_dist_{distance}_loss_{loss_type}_dim_{latent_dim}_lr{lr}'
    #
    #     try:
    #         avg_davies, avg_silhouette = main(config, params, use_legacy)
    #
    #         # Store results
    #         result = {
    #             'params': current_params,
    #             'davies_score': avg_davies,
    #             'silhouette_score': avg_silhouette
    #         }
    #         all_results.append(result)
    #
    #         # Update best results
    #         if avg_silhouette > best_results['silhouette']['score']:
    #             best_results['silhouette']['score'] = avg_silhouette
    #             best_results['silhouette']['params'] = current_params
    #             print(f"\nNew best silhouette score: {avg_silhouette:.4f}")
    #
    #         if avg_davies < best_results['davies']['score']:  # Lower is better for Davies-Bouldin
    #             best_results['davies']['score'] = avg_davies
    #             best_results['davies']['params'] = current_params
    #             print(f"\nNew best Davies-Bouldin score: {avg_davies:.4f}")
    #
    #         # Save current state after each experiment
    #         def json_serialize(obj):
    #             """Convert numpy types to Python native types for JSON serialization"""
    #             if isinstance(obj, np.integer):
    #                 return int(obj)
    #             elif isinstance(obj, np.floating):
    #                 return float(obj)
    #             elif isinstance(obj, np.ndarray):
    #                 return obj.tolist()
    #             else:
    #                 return obj
    #
    #
    #         # When saving results:
    #         with open('grid_search_results.json', 'w') as f:
    #             results_dict = {
    #                 'best_results': {
    #                     'silhouette': {
    #                         'score': float(best_results['silhouette']['score']),
    #                         'params': {k: json_serialize(v) for k, v in best_results['silhouette']['params'].items()}
    #                     },
    #                     'davies': {
    #                         'score': float(best_results['davies']['score']),
    #                         'params': {k: json_serialize(v) for k, v in best_results['davies']['params'].items()}
    #                     }
    #                 },
    #                 'all_results': [{
    #                     'params': {k: json_serialize(v) for k, v in result['params'].items()},
    #                     'davies_score': float(result['davies_score']),
    #                     'silhouette_score': float(result['silhouette_score'])
    #                 } for result in all_results]
    #             }
    #             json.dump(results_dict, f, indent=4)
    #
    #     except Exception as e:
    #         print(f"Error in experiment: {str(e)}")
    #         continue
    #
    # print("\nGrid search completed!")
    #
    # import itertools
    # from copy import deepcopy
    # import json
    # import numpy as np
    #
    # # Define parameter grids
    # grids = {
    #     'window_size': [50],
    #     'batch_size': [128],
    #     'ae_epochs': [100],
    #     'depth': [3],
    #     'num_filters': [4],
    #     'mask_probability': [0.3],
    #     'margin': [0.1, 0.3],
    #     'distance_metric': ['cosine', 'L2'],
    #     'triplet_weight': [0, 10,50],
    #     'lr': [0.001,0.1]
    # }
    #
    #
    # # Base parameters that don't change
    # base_params = {
    #     'name': 'old data with trip grid',
    #     'resample': False,
    #     'classification_epochs': 100,
    #     'early_stopping_patience': 10,
    #     'threshold': 0.9,
    #     'use_gan': False,
    #     'TRAIN': True,
    #     'classifier': 'add dropout =0.1',
    #     'participant':1
    # }
    #
    # # Track best results
    # best_results = {
    #     'silhouette': {'score': float('-inf'), 'params': None, 'vae_params': None},
    #     'davies': {'score': float('inf'), 'params': None, 'vae_params': None}
    # }
    # all_results = []
    #
    # # Load legacy data once
    # df = load_eye_tracking_data(
    #     data_path=config.data_path,
    #     approach_num=config.approach_num,
    #     participant_id=config.participant_id,
    #     data_format="legacy"
    # )
    #
    # # Calculate total experiments
    # param_combinations = list(itertools.product(
    #     *[grids[param] for param in grids.keys()]
    # ))
    # total_experiments = len(param_combinations)
    # current_experiment = 0
    #
    # # Run grid search
    # for values in param_combinations:
    #     current_experiment += 1
    #
    #     # Split values into main params and vae params
    #     main_param_values = values[:len(grids)]
    #     vae_param_values = values[len(grids):]
    #
    #     # Create parameter dictionaries
    #     current_params = base_params.copy()
    #     current_vae_params = {}
    #
    #     # Update main parameters
    #     for param, value in zip(grids.keys(), main_param_values):
    #         current_params[param] = value
    #
    #
    #     print(f"\nExperiment {current_experiment}/{total_experiments}")
    #     print("\nMain parameters:")
    #
    #
    #
    #     try:
    #         # Create method name
    #         method_name = create_method_name(current_params['name'], config, {**current_params, **current_vae_params})
    #         davies_score, silhouette_score = main_with_autoencoder(
    #             df=df,
    #             window_size=current_params['window_size'],
    #             method=method_name,
    #             resample=current_params['resample'],
    #             classification_epochs=current_params['classification_epochs'],
    #             batch_size=current_params['batch_size'],
    #             ae_epochs=current_params['ae_epochs'],
    #             depth=current_params['depth'],
    #             num_filters=current_params['num_filters'],
    #             lr=current_params['lr'],
    #             mask_probability=current_params['mask_probability'],
    #             early_stopping_patience=current_params['early_stopping_patience'],
    #             threshold=current_params['threshold'],
    #             use_gan=current_params['use_gan'],
    #             TRAIN=current_params['TRAIN'],
    #             use_vae=False,
    #             vae_params=None
    #         )
    #
    #         # Store results
    #         result = {
    #             'params': current_params.copy(),
    #             'vae_params': current_vae_params.copy(),
    #             'davies_score': davies_score,
    #             'silhouette_score': silhouette_score
    #         }
    #         all_results.append(result)
    #
    #         # Update best results
    #         if silhouette_score > best_results['silhouette']['score']:
    #             best_results['silhouette'].update({
    #                 'score': silhouette_score,
    #                 'params': current_params.copy(),
    #                 'vae_params': current_vae_params.copy()
    #             })
    #             print(f"\nNew best silhouette score: {silhouette_score:.4f}")
    #
    #         if davies_score < best_results['davies']['score']:
    #             best_results['davies'].update({
    #                 'score': davies_score,
    #                 'params': current_params.copy(),
    #                 'vae_params': current_vae_params.copy()
    #             })
    #             print(f"\nNew best Davies-Bouldin score: {davies_score:.4f}")
    #
    #         # Save results after each experiment
    #         with open('vae_legacy_grid_search_results.json', 'w') as f:
    #             json.dump({
    #                 'best_results': best_results,
    #                 'all_results': all_results
    #             }, f, indent=4, default=lambda x: float(x) if isinstance(x, np.number) else x)
    #
    #     except Exception as e:
    #         print(f"Error in experiment: {str(e)}")
    #         continue
    #
    # print("\nGrid search completed!")
    # print("\nBest Results:")
    # print("\nBest Silhouette Score:", best_results['silhouette']['score'])
    # print("Parameters:", best_results['silhouette']['params'])
    # print("VAE Parameters:", best_results['silhouette']['vae_params'])
    # print("\nBest Davies Score:", best_results['davies']['score'])
    # print("Parameters:", best_results['davies']['params'])
    # print("VAE Parameters:", best_results['davies']['vae_params'])
    #
# [[11958   140]
#  [  128    11]]
#
# Test Set Metrics:
# precision: 0.0728
# recall: 0.0791
# f1: 0.0759
# minority_precision: 0.0728
# minority_recall: 0.0791
# minority_f1: 0.0759
# {'precision': 0.0728476821192053, 'recall': 0.07913669064748201, 'f1': 0.07586206896551724, 'minority_precision': 0.0728476821192053, 'minority_recall': 0.07913669064748201, 'minority_f1': 0.07586206896551724}
# Training Progress:  26%|▎| 10/39 [00:07<00:21,  1.36it/s, train_loss=0.5210, tra
# Early stopping triggered at epoch 11
# Training Progress:  26%|▎| 10/39 [00:08<00:23,  1.23it/s, train_loss=0.5210, tra
#
# Finding optimal threshold...
# Collecting predictions: 100%|███████████████| 285/285 [00:00<00:00, 3658.47it/s]
# Finding optimal threshold: 100%|███████████| 100/100 [00:00<00:00, 25503.49it/s]
# No threshold satisfies the criteria. Consider relaxing constraints.
# Evaluating: 100%|███████████████████████████| 383/383 [00:00<00:00, 3719.29it/s]
#               precision    recall  f1-score   support
#
#            0       0.97      0.22      0.36     12098
#            1       0.00      0.32      0.01       139
#
#     accuracy                           0.22     12237
#    macro avg       0.48      0.27      0.18     12237
# weighted avg       0.95      0.22      0.35     12237
#
# [[2639 9459]
#  [  95   44]]