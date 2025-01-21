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

from representation_method.utils.trainers import FocalLoss, EnsembleTrainer
from representation_method.utils.visualization import analyze_embeddings_from_loader, \
    analyze_vae_embeddings_from_loader, analyze_vae_embeddings_with_umap, analyze_encoder_embeddings_with_umap_2
# Import our utility modules
from utils.general_utils import seed_everything
from utils.data_utils import create_time_series, split_train_test_for_time_series, DataSplitter, \
    create_dynamic_time_series, find_max_consecutive_hits, create_dynamic_time_series_for_rf, \
    create_dynamic_time_series_with_smooth_labels
from utils.losses import WeightedCrossEntropyLoss, ContrastiveAutoencoderLoss, ImbalancedTripletContrastiveLoss, \
    CombinedTemporalLoss
from utils.gan_utils import generate_balanced_data_with_gan
from utils.data_loader import load_eye_tracking_data, DataConfig, create_data_loader
from models.autoencoder import CNNRecurrentAutoencoder, initialize_weights, TimeSeriesVAE, ImprovedTimeSeriesVAE, \
    CombinedVAEClassifier
from models.classifier import CombinedModel, TVAEClassifier, EnhancedClassifier, EnhancedCombinedModel, \
    SimpleCombinedModel, ComplexCNNClassifier
from utils.trainers import AutoencoderTrainer, CombinedModelTrainer, ContrastiveAutoencoderTrainer, VAETrainer, \
    VAEClassifierTrainer, ImprovedVAEClassifierTrainer, TripletAutoencoderTrainer
from latent_space_classifer import  LatentSpaceClassifier
feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X', 'CURRENT_FIX_IA_Y',
                   'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT']
from sklearn.ensemble import RandomForestClassifier


def filter_easy_negatives(X, y, recall_target=0.95, max_depth=6, n_estimators=50):
    """
    Train a simple random forest to filter out 'easy' negative windows
    while retaining ~95% recall for positives.

    Args:
        X (np.ndarray): shape (num_windows, window_size, num_features) or (num_windows, feature_dim).
        y (np.ndarray): shape (num_windows,).
        recall_target (float): Desired recall for positive class (0.0 - 1.0).
        max_depth (int): RandomForest max_depth parameter.
        n_estimators (int): RandomForest n_estimators parameter.

    Returns:
        keep_mask (np.ndarray): Boolean mask of windows to KEEP (True).
        model (RandomForestClassifier): Trained RF model (for reference).
        chosen_threshold (float): Probability threshold used.
    """
    # 1) Flatten each window so the coarse model sees a simpler 1D feature vector per window
    #    (Alternatively, you could extract some aggregated features instead of flattening.)
    num_windows = X.shape[0]
    X_flat = X.reshape(num_windows, -1)

    # 2) Train a simple RandomForest on (X_flat, y)
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',  # helps with imbalance
        random_state=42
    )
    rf.fit(X_flat, y)

    # 3) Get predicted probabilities for the positive class
    pos_probs = rf.predict_proba(X_flat)[:, 1]

    # 4) Choose a threshold that yields the desired recall
    #    Sort windows by descending probability
    sorted_indices = np.argsort(-pos_probs)
    pos_count = np.sum(y == 1)
    found_positives = 0
    chosen_threshold = 0.0

    for idx in sorted_indices:
        if y[idx] == 1:
            found_positives += 1
        current_recall = found_positives / pos_count if pos_count > 0 else 1.0
        if current_recall >= recall_target:
            chosen_threshold = pos_probs[idx]
            break

    # 5) Create mask of windows with probability >= chosen_threshold
    keep_mask = pos_probs >= chosen_threshold

    print(f"\n[Coarse Filter] Achieved ~{recall_target * 100:.1f}% recall")
    print(f"[Coarse Filter] Probability threshold chosen = {chosen_threshold:.4f}")
    print(f"[Coarse Filter] Keeping {np.sum(keep_mask)}/{num_windows} windows (~{100 * np.mean(keep_mask):.1f}%)")

    return keep_mask, rf, chosen_threshold

def evaluate_random_classifier(Y_test,method_dir = None):
    """
    Evaluate a random classifier against the given Y_test.
    """
    # Calculate class distribution
    class_counts = np.bincount(Y_test)
    class_probs = class_counts / len(Y_test)

    # Generate random predictions based on class probabilities
    random_predictions = np.random.choice(len(class_probs), size=len(Y_test), p=class_probs)

    # Calculate metrics
    acc = accuracy_score(Y_test, random_predictions)
    precision = precision_score(Y_test, random_predictions, average="binary", pos_label=1)
    recall = recall_score(Y_test, random_predictions, average="binary", pos_label=1)
    f1 = f1_score(Y_test, random_predictions, average="binary", pos_label=1)

    # Print classification report
    print("Random Classifier Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    if method_dir:
        os.makedirs(method_dir, exist_ok=True)
        results_path = os.path.join(method_dir, "random_classifier_metrics.json")
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {results_path}")

    # Return metrics
    return metrics

# feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION']
def main_with_autoencoder(df, window_size=5, method='', resample=False, classification_epochs=20, batch_size=32,
                          ae_epochs=100, depth=4, num_filters=32, lr=0.001, mask_probability=0.4,
                          early_stopping_patience=30, threshold=0.5, use_gan=False, TRAIN = False, use_vae=True,
                          vae_params=None, assemble=False,seed = 0):
    """
    Main training pipeline with autoencoder and optional GAN augmentation.
    """
    seed_everything(seed)
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

    print(f"window_size is {window_size}")
    train_df, test_df = split_train_test_for_time_series(df, test_size=0.2, random_state=seed)
    train_df, val_df = split_train_test_for_time_series(train_df, test_size=0.2, random_state=seed)

    print("Original class distribution in test set:")
    print(test_df["target"].value_counts())
    # Create time series
    X_train, Y_train ,_= create_dynamic_time_series(
        train_df,feature_columns=None,participant_id=config.participant_id,load_existing=False,split_type='train',window_size=window_size)
    X_test, Y_test,_ = create_dynamic_time_series(
        test_df,feature_columns=None,participant_id=config.participant_id,load_existing=False,split_type='test',window_size=window_size)
    X_val, Y_val,_ = create_dynamic_time_series(
        val_df,feature_columns=None,participant_id=config.participant_id,load_existing=False,split_type='test',window_size=window_size)

    # evaluate_random_classifier(Y_test,method_dir)
    if window_size == 10:
        depth = 3
    elif window_size in [20,50,150]:
        depth = 4
    else:
        depth = 2

    # print("after window class distribution in test set:")
    # print(pd.Series(Y_test).value_counts())
    # Create time series

    # results, best_clf = analyze_dynamic_windows(X_train, Y_train, X_test, Y_test, X_val, Y_val)
    # return results
    # Scale data
    # print("Before first filtering")
    # print(pd.Series(Y_train).value_counts())
    # keep_mask, rf_model, coarse_threshold = filter_easy_negatives(
    #     X_train, Y_train,
    #     recall_target=0.99,
    #     max_depth=6,
    #     n_estimators=50
    # )
    #
    # # Apply mask to remove easy negatives from training data
    # # X_train = X_train[keep_mask]
    # # Y_train = Y_train[keep_mask]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    input_dim = X_train.shape[-1]
    if use_vae:
        autoencoder = TimeSeriesVAE(
            input_dim=input_dim,
            hidden_dim=64,
            latent_dim=vae_params.get('latent_dim', 32)
        ).to(device)

        # autoencoder = ImprovedTimeSeriesVAE(len(feature_columns)+2,latent_dim=vae_params.get('latent_dim', 32)).to(device)


    else:
        autoencoder =CNNRecurrentAutoencoder(
            in_channels=input_dim,  # Number of input features
            num_filters=32,  # Match the old model's filter count
            depth=depth ,  # - original 4
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

        Y_train_1d = Y_train.flatten() if isinstance(Y_train, np.ndarray) else np.array(Y_train).flatten()
        Y_val_1d = Y_val.flatten() if isinstance(Y_val, np.ndarray) else np.array(Y_val).flatten()
        Y_test_1d = Y_test.flatten() if isinstance(Y_test, np.ndarray) else np.array(Y_test).flatten()

        print("Shape after flattening:", Y_train_1d.shape)  # Debug print

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
            # trainer = TripletAutoencoderTrainer(
            #     model=autoencoder,
            #     criterion=nn.MSELoss(),
            #     optimizer=optimizer,
            #     scheduler=scheduler,
            #     device=device,
            #     margin=params.get('margin', 10),
            #     triplet_weight=params.get('triplet_weight', 30),
            #     distance_metric=params.get('distance_metric', 'cosine'),
            #     mask_probability=params.get('mask_probability', 0.1),
            #     save_path=method_dir,
            #     early_stopping_patience=early_stopping_patience
            #
            #     )
            trainer = AutoencoderTrainer(
                model=autoencoder,
                criterion=nn.MSELoss(),
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                mask_probability=params.get('mask_probability', 0.4),
                save_path=method_dir,
                early_stopping_patience=early_stopping_patience,
                recon_criterion=nn.MSELoss(),
                cls_criterion= FocalLoss()

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
        best_autoencoder_path = os.path.join(root_dir, "results",'add focal loss to AE _approach_6_participant_1use_gan_False',
                                             'best_autoencoder_model.pth')


        # autoencoder.load_state_dict(torch.load(best_autoencoder_path))

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    ensemble_save_path = os.path.join(method_dir,'ensemble_models')
    os.makedirs(ensemble_save_path, exist_ok=True)
    input_dim = X_train_tensor.shape[1]

    if assemble:
        ensemble_trainer = EnsembleTrainer(
            base_model_class=ComplexCNNClassifier,
            model_params={'input_dim': input_dim},
            n_models=10,
            device='cuda',
            save_path=ensemble_save_path

        )

        # Train ensemble

        ensemble_trainer.train_ensemble(
            train_dataset=train_dataset,
            val_loader=val_loader,
            batch_size=32,
            epochs=50,
            criterion = nn.CrossEntropyLoss(),
            optimizer_class=optim.Adam,
            optimizer_params={'lr': 0.001},majority_weight=weights[0]
        )
        accuracy, f1= ensemble_trainer.evaluate(test_loader)
        return accuracy,f1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    model = SimpleCombinedModel(autoencoder, num_classes=2).to(device)
    # TVAEClassifier
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': lr },
        {'params': model.classifier.parameters(), 'lr': lr}
    ], weight_decay=1e-4)

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
        f'seed_{params["seed"]}'
        f'window_size_{params["window_size"]}'
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
        'TRAIN': False,
        "classifier": "add dropout =0.1",
        "margin":0.1,
        "distance_metric":'cosine',
        "triplet_weight":0.5,
        'lr':0.0001,
        'latent_dim':32,
    }



    # Load legacy data
    df = load_eye_tracking_data(
        data_path=config.data_path,
        approach_num=config.approach_num,
        participant_id=config.participant_id,
        data_format="legacy"
    )

    # Combine params for method name SimpleCombinedModel,150,200,300,500
    combined_params = params  # This merges both dictionaries
    for window in [100]:
        print(f"####### WINDOW {window}#########")
        try:
            # for seed in  [0,5,13, 123, 999, 2025]:
            for seed in  [0]:
                print(f"#######SEED {seed}#########")
                combined_params["seed"] = seed
                params['window_size'] = window
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
                    seed=seed
                )
        except Exception as e:
            print(e)
            continue


# [[4326 4593]
#  [1189 1337]]
# ansemble
# [[3304 5615]
#  [ 953 1573]]

# [[5624 3295]
#  [1557  969]]

# [[3312 5607]
#  [ 868 1658]]


#########################
#
# Ensemble Results:
# Accuracy: 0.5966
# F1 Score: 0.6267
# acc
# 0.5965923984272609
# minority_precision
# 0.2563505010487066
# minority_recall
# 0.43547110055423593
# minority_f1
# 0.3227226052515769
# [[5728 3191]
#  [1426 1100]]


# with GRU improved little bit - recal to 0.5 m precison to 0.24