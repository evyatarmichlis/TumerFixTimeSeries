import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Union
import logging
from PatchTST.PatchTST_self_supervised.src.models.patchTST import  PatchTSTEncoder, ClassificationHead

import torch.nn.functional as F
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import logging
import time
from tqdm import tqdm
import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from representation_method.utils.losses import FocalLoss
from transformer_method.models.triplet_loss import TripletLoss


class PatchTSTClassifier(nn.Module):

    """
    PatchTST model adapted for classification tasks.
    Extends the original PatchTST architecture while maintaining compatibility
    with pretrained weights and allowing for customization.
    """

    def __init__(
            self,
            c_in: int,
            patch_len: int,
            stride: int,
            num_patch: int,
            n_classes: int = 2,
            d_model: int = 128,
            n_heads: int = 16,
            n_layers: int = 3,
            d_ff: int = 256,
            dropout: float = 0.5,
            head_dropout: float = 0.5,
            attn_dropout: float = 0.5,
            act: str = "gelu",
            shared_embedding: bool = True,
            norm: str = 'BatchNorm',
            pe: str = 'zeros',
            learn_pe: bool = True,
            verbose: bool = False,
            **kwargs
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = num_patch
        self.backbone = PatchTSTEncoder(
            c_in=c_in,
            num_patch=num_patch,
            patch_len=patch_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            shared_embedding=shared_embedding,
            d_ff=d_ff,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            res_attention=True,
            pre_norm=True,
            store_attn=False,
            pe=pe,
            learn_pe=learn_pe,
            verbose=verbose,
            **kwargs
        )

        self.head = ClassificationHead(
            n_vars=c_in,
            d_model=d_model,
            n_classes=n_classes,
            head_dropout=head_dropout
        )

        # Extra attributes for model info
        self.n_params = sum(p.numel() for p in self.parameters())
        self.config = {
            'c_in': c_in,
            'patch_len': patch_len,
            'stride': stride,
            'num_patch': num_patch,
            'n_classes': n_classes,
            'd_model': d_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'd_ff': d_ff,
            'dropout': dropout,
            'head_dropout': head_dropout,
            'attn_dropout': attn_dropout,
            'act': act,
            'shared_embedding': shared_embedding,
            'norm': norm,
            'pe': pe,
            'learn_pe': learn_pe
        }

        self._setup_logging()
        if verbose:
            self._log_model_info()

    def _setup_logging(self):
        """Setup logging for the model"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())
            self.logger.setLevel(logging.INFO)

    def _log_model_info(self):
        """Log model configuration and parameters"""
        self.logger.info(f"\nPatchTSTClassifier Configuration:")
        self.logger.info(f"Input Features: {self.config['c_in']}")
        self.logger.info(f"Number of Classes: {self.config['n_classes']}")
        self.logger.info(f"Patch Length: {self.config['patch_len']}")
        self.logger.info(f"Number of Patches: {self.config['num_patch']}")
        self.logger.info(f"Model Dimension: {self.config['d_model']}")
        self.logger.info(f"Number of Layers: {self.config['n_layers']}")
        self.logger.info(f"Number of Attention Heads: {self.config['n_heads']}")
        self.logger.info(f"Total Parameters: {self.n_params:,}")

    def forward(self, z, chunk_size=1, seq_chunk_size=None):
        """
        Process input data in chunks to reduce memory usage.
        """
        # Get initial shape
        bs, num_patch, patch_len, n_vars = z.shape
        z = z.permute(0, 1, 3, 2)  # [bs, num_patch, n_vars, patch_len]

        if (chunk_size is None or bs <= chunk_size) and (seq_chunk_size is None or num_patch <= seq_chunk_size):
            # Process entire input through backbone and head
            with torch.cuda.amp.autocast(enabled=True):
                z = self.backbone(z)  # Backbone expects [bs x num_patch x n_vars x patch_len]
                z = self.head(z)
            return z

        outputs = []
        # Process in batch chunks
        for batch_start in range(0, bs, chunk_size or bs):
            batch_end = min(batch_start + chunk_size, bs)
            batch_chunk = z[batch_start:batch_end]  # [chunk_size, num_patch, n_vars, patch_len]

            if seq_chunk_size is not None and num_patch > seq_chunk_size:
                accumulated = None

                # Process sequence chunks
                for seq_start in range(0, num_patch, seq_chunk_size):
                    seq_end = min(seq_start + seq_chunk_size, num_patch)
                    seq_chunk = batch_chunk[:, seq_start:seq_end]  # [chunk_size, seq_chunk_size, n_vars, patch_len]

                    with torch.cuda.amp.autocast(enabled=True):
                        chunk_output = self.backbone(seq_chunk)

                        if accumulated is None:
                            # Match backbone output dimensions
                            accumulated = torch.zeros(
                                batch_end - batch_start,  # chunk_size
                                n_vars,
                                chunk_output.size(2),  # d_model
                                num_patch,
                                device=chunk_output.device
                            )

                        accumulated[:, :, :, seq_start:seq_end] = chunk_output

                    torch.cuda.empty_cache()

                with torch.cuda.amp.autocast(enabled=True):
                    batch_output = self.head(accumulated)
                    outputs.append(batch_output)

            else:
                with torch.cuda.amp.autocast(enabled=True):
                    batch_output = self.backbone(batch_chunk)
                    batch_output = self.head(batch_output)
                    outputs.append(batch_output)

            torch.cuda.empty_cache()

        return torch.cat(outputs, dim=0)

    def get_attention_maps(self) -> Optional[torch.Tensor]:
        """
        Get attention maps from the backbone if available.

        Returns:
            Attention maps tensor if store_attn was True, None otherwise
        """
        if hasattr(self.backbone, 'get_attention_maps'):
            return self.backbone.get_attention_maps()
        return None

    def freeze_backbone(self):
        """Freeze the backbone parameters for fine-tuning only the head"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.logger.info("Backbone parameters frozen")

    def unfreeze_backbone(self):
        """Unfreeze the backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.logger.info("Backbone parameters unfrozen")

    def get_config(self) -> Dict:
        """Get model configuration"""
        return self.config


    def train_model(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            test_loader: Optional[DataLoader] = None,
            epochs: int = 50,
            learning_rate: float = 0.001,
            weight_decay: float = 0.01,
            patience: int = 10,
            output_dir: Optional[Path] = None,
            device: Optional[torch.device] = None,
            class_weights: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Train the PatchTST classifier.

        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            test_loader: Optional DataLoader for test data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            patience: Early stopping patience
            output_dir: Directory to save checkpoints and logs
            device: Device to train on
            class_weights: Optional tensor of class weights for imbalanced data

        Returns:
            Dictionary containing training history
        """
        # Setup
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # if class_weights is not None:
        #     criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        # else:
        #     criterion = nn.CrossEntropyLoss()
        criterion = FocalLoss(alpha=class_weights[1], gamma=2).to(device)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=epochs // 3,
            T_mult=2
        )

        # Training history
        history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rates': []
        }

        # Early stopping setup
        best_val_loss = float('inf')
        best_epoch = 0
        best_model = None
        no_improve = 0

        self.logger.info(f"\nStarting training on {device}")
        self.logger.info(f"Total epochs: {epochs}")
        self.logger.info(f"Learning rate: {learning_rate}")
        self.logger.info(f"Weight decay: {weight_decay}")

        try:
            for epoch in range(epochs):
                epoch_start = time.time()

                # Training
                train_metrics = self._train_epoch(
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    epoch,
                    epochs
                )

                # Validation
                if val_loader is not None:
                    val_metrics = self._validate(val_loader, criterion, device)

                    # Early stopping check
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        best_epoch = epoch
                        best_model = self.state_dict()
                        no_improve = 0

                        if output_dir:
                            self._save_checkpoint(
                                output_dir / 'best_model.pth',
                                epoch,
                                optimizer,
                                val_metrics
                            )
                    else:
                        no_improve += 1

                # Update learning rate
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]

                # Update history
                history['train_loss'].append(train_metrics['loss'])
                history['train_acc'].append(train_metrics['acc'])
                history['train_f1'].append(train_metrics['f1'])
                history['learning_rates'].append(current_lr)

                if val_loader is not None:
                    history['val_loss'].append(val_metrics['loss'])
                    history['val_acc'].append(val_metrics['acc'])
                    history['val_f1'].append(val_metrics['f1'])

                # Log progress
                epoch_time = time.time() - epoch_start
                self._log_epoch(epoch, epochs, train_metrics, val_metrics if val_loader else None,
                                current_lr, epoch_time)

                # Save current state
                if output_dir:
                    self._save_checkpoint(
                        output_dir / 'last_model.pth',
                        epoch,
                        optimizer,
                        val_metrics if val_loader else train_metrics
                    )
                    self._save_history(history, output_dir / 'history.json')

                # Early stopping
                if no_improve >= patience:
                    self.logger.info(f"\nEarly stopping triggered! No improvement for {patience} epochs")
                    break

        except KeyboardInterrupt:
            self.logger.info("\nTraining interrupted by user!")

        # Load best model if available
        if best_model is not None:
            self.load_state_dict(best_model)
            self.logger.info(f"\nLoaded best model from epoch {best_epoch}")

        # Final evaluation on test set
        if test_loader is not None:
            test_metrics = self._validate(test_loader, criterion, device)
            self.logger.info("\nTest Set Results:")
            # self._log_metrics(test_metrics)
            history['test_metrics'] = test_metrics

        # Plot training history
        if output_dir:
            self._plot_training_history(history, output_dir)

        return history

    def _train_epoch(
            self,
            train_loader: DataLoader,
            criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            epoch: int,
            total_epochs: int
    ) -> Dict:
        """Train for one epoch with enhanced stability measures."""
        self.train()
        metrics = {
            'loss': 0.0,
            'correct': 0,
            'total': 0,
            'predictions': [],
            'targets': []
        }

        # Initialize gradient scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        # Set chunk size based on available GPU memory
        total_memory = torch.cuda.get_device_properties(device).total_memory
        chunk_size = min(4, train_loader.batch_size)

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epochs}',
                    leave=False, dynamic_ncols=True)

        for batch_idx, (inputs, targets) in enumerate(pbar):
            try:
                # Move data to device efficiently
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                # Clear gradients
                optimizer.zero_grad(set_to_none=True)

                # Process in chunks if batch size is large
                if inputs.size(0) > chunk_size:
                    outputs = []
                    loss = 0.0

                    for chunk_start in range(0, inputs.size(0), chunk_size):
                        chunk_end = min(chunk_start + chunk_size, inputs.size(0))
                        input_chunk = inputs[chunk_start:chunk_end]
                        target_chunk = targets[chunk_start:chunk_end]

                        # Forward pass with mixed precision
                        with torch.cuda.amp.autocast():
                            chunk_output = self(input_chunk)
                            chunk_loss = criterion(chunk_output, target_chunk)
                            loss += chunk_loss * (chunk_end - chunk_start) / inputs.size(0)
                            outputs.append(chunk_output)

                        # Clear cache after each chunk
                        torch.cuda.empty_cache()

                    outputs = torch.cat(outputs, dim=0)
                else:
                    # Forward pass for small batches
                    with torch.cuda.amp.autocast():
                        outputs = self(inputs)
                        loss = criterion(outputs, targets)

                # Check for invalid loss
                if not torch.isfinite(loss):
                    self.logger.warning(f"Warning: Loss is {loss.item()}, skipping batch")
                    continue

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Unscale gradients for clipping
                scaler.unscale_(optimizer)

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                # Optimizer step with gradient scaling
                scaler.step(optimizer)
                scaler.update()

                # Update metrics (outside of autocast for accuracy)
                with torch.no_grad():
                    metrics['loss'] += loss.item() * targets.size(0)
                    _, predicted = outputs.max(1)
                    metrics['total'] += targets.size(0)
                    metrics['correct'] += predicted.eq(targets).sum().item()

                    # Move predictions to CPU to save GPU memory
                    metrics['predictions'].extend(predicted.cpu().numpy())
                    metrics['targets'].extend(targets.cpu().numpy())

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100. * metrics['correct'] / metrics['total']:.2f}%",
                    'chunk': f"{chunk_size}" if inputs.size(0) > chunk_size else "full"
                })

                # Adaptively adjust chunk size based on GPU memory usage
                if batch_idx % 10 == 0:
                    memory_used = torch.cuda.memory_allocated(device)
                    memory_ratio = memory_used / total_memory

                    if memory_ratio > 0.85 and chunk_size > 8:
                        chunk_size = max(8, chunk_size // 2)
                    elif memory_ratio < 0.5 and chunk_size < train_loader.batch_size:
                        chunk_size = min(train_loader.batch_size, chunk_size * 2)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Handle OOM by reducing chunk size and clearing cache
                    torch.cuda.empty_cache()
                    chunk_size = max(4, chunk_size // 2)
                    self.logger.warning(f"GPU OOM in batch {batch_idx}, reduced chunk size to {chunk_size}")
                    if chunk_size < 4:
                        raise e  # If chunk size gets too small, raise the error
                    continue
                else:
                    raise e

        # Calculate final metrics
        return self._calculate_metrics(metrics, len(train_loader), 'Train')

    def _validate(
            self,
            dataloader: DataLoader,
            criterion: nn.Module,
            device: torch.device
    ) -> Dict:
        """Validate the model"""
        self.eval()
        metrics = {
            'loss': 0,
            'correct': 0,
            'total': 0,
            'predictions': [],
            'targets': []
        }

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, targets)

                metrics['loss'] += loss.item()
                _, predicted = outputs.max(1)
                metrics['total'] += targets.size(0)
                metrics['correct'] += predicted.eq(targets).sum().item()

                metrics['predictions'].extend(predicted.cpu().numpy())
                metrics['targets'].extend(targets.cpu().numpy())

        return self._calculate_metrics(metrics, len(dataloader),'Validation')

    def plot_confusion_matrix(self,y_true, y_pred):
        """Plots and saves the confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        # Print the confusion matrix in a readable format
        print("Confusion Matrix:")
        print("   Predicted 0  Predicted 1")
        print(f"Actual 0   {cm[0, 0]:<10} {cm[0, 1]:<10}")
        print(f"Actual 1   {cm[1, 0]:<10} {cm[1, 1]:<10}")

    def _calculate_metrics(self, metrics: Dict, num_batches: int,dataset_name = 'Train') -> Dict:
        """Calculate classification metrics including per-class precision and recall"""
        predictions = np.array(metrics['predictions'])
        targets = np.array(metrics['targets'])

        # Calculate confusion matrix first
        cm = confusion_matrix(targets, predictions)
        print(f"{dataset_name} Confusion Matrix:")
        self.plot_confusion_matrix(targets, predictions)

        # Calculate per-class metrics
        precision_per_class = precision_score(targets, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(targets, predictions, average=None, zero_division=0)

        return {
            'loss': metrics['loss'] / num_batches,
            'acc': 100. * metrics['correct'] / metrics['total'],
            'precision': precision_score(targets, predictions, average='weighted', zero_division=0),
            'recall': recall_score(targets, predictions, average='weighted', zero_division=0),
            'f1': f1_score(targets, predictions, average='weighted', zero_division=0),
            'confusion_matrix': cm,
            'precision_0': precision_per_class[0],
            'precision_1': precision_per_class[1],
            'recall_0': recall_per_class[0],
            'recall_1': recall_per_class[1]
        }

    def _log_epoch(
            self,
            epoch: int,
            total_epochs: int,
            train_metrics: Dict,
            val_metrics: Optional[Dict],
            lr: float,
            epoch_time: float
    ):
        """Log epoch results with per-class metrics"""
        self.logger.info(
            f"\nEpoch {epoch + 1}/{total_epochs} - {epoch_time:.2f}s - LR: {lr:.6f}"
        )
        self.logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Acc: {train_metrics['acc']:.2f}%, "
            f"F1: {train_metrics['f1']:.4f}, "
            f"Recall-0: {train_metrics['recall_0']:.4f}, "
            f"Recall-1: {train_metrics['recall_1']:.4f}, "
            f"Precision-0: {train_metrics['precision_0']:.4f}, "
            f"Precision-1: {train_metrics['precision_1']:.4f}"
        )
        if val_metrics:
            self.logger.info(
                f"Val - Loss: {val_metrics['loss']:.4f}, "
                f"Acc: {val_metrics['acc']:.2f}%, "
                f"F1: {val_metrics['f1']:.4f}, "
                f"Recall-0: {val_metrics['recall_0']:.4f}, "
                f"Recall-1: {val_metrics['recall_1']:.4f}, "
                f"Precision-0: {val_metrics['precision_0']:.4f}, "
                f"Precision-1: {val_metrics['precision_1']:.4f}"
            )

    def _save_checkpoint(
            self,
            path: Path,
            epoch: int,
            optimizer: torch.optim.Optimizer,
            metrics: Dict
    ):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, path)

    def _save_history(self, history: Dict, path: Path):
        """Save training history"""
        with open(path, 'w') as f:
            json.dump(history, f, indent=4)

    def _log_epoch(
            self,
            epoch: int,
            total_epochs: int,
            train_metrics: Dict,
            val_metrics: Optional[Dict],
            lr: float,
            epoch_time: float
    ):
        """Log epoch results"""
        self.logger.info(
            f"\nEpoch {epoch + 1}/{total_epochs} - {epoch_time:.2f}s - LR: {lr:.6f}"
        )
        self.logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Acc: {train_metrics['acc']:.2f}%, "
            f"F1: {train_metrics['f1']:.4f}"
            f"Recall-0: {train_metrics['recall_0']:.4f}, "
            f"Recall-1: {train_metrics['recall_1']:.4f}, "
            f"Precision-0: {train_metrics['precision_0']:.4f}, "
            f"Precision-1: {train_metrics['precision_1']:.4f}"
        )
        if val_metrics:
            self.logger.info(
                f"Val - Loss: {val_metrics['loss']:.4f}, "
                f"Acc: {val_metrics['acc']:.2f}%, "
                f"F1: {val_metrics['f1']:.4f}"
                f"Recall-0: {val_metrics['recall_0']:.4f}, "
                f"Recall-1: {val_metrics['recall_1']:.4f}, "
                f"Precision-0: {val_metrics['precision_0']:.4f}, "
                f"Precision-1: {val_metrics['precision_1']:.4f}"
            )

    def _plot_training_history(self, history: Dict, output_dir: Path):
        """Plot and save training history"""
        metrics = ['loss', 'acc', 'f1']
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, metric in enumerate(metrics):
            axes[idx].plot(history[f'train_{metric}'], label=f'Train {metric}')
            if f'val_{metric}' in history:
                axes[idx].plot(history[f'val_{metric}'], label=f'Val {metric}')
            axes[idx].set_title(f'{metric.capitalize()} History')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel(metric.capitalize())
            axes[idx].legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'training_history.png')
        plt.close()


    def load_pretrained(self, state_dict: Dict[str, torch.Tensor], logger: Optional[logging.Logger] = None) -> None:
        """
        Load pretrained weights with support for partial loading.

        Args:
            state_dict: Dictionary containing pretrained weights
            logger: Optional logger for detailed loading information
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        # Get current model state dict
        model_state = self.state_dict()

        # Filter and load matched layers
        loaded_keys = []
        skipped_keys = []

        for name, param in state_dict.items():
            if name in model_state:
                if model_state[name].shape == param.shape:
                    model_state[name].copy_(param)
                    loaded_keys.append(name)
                else:
                    skipped_keys.append((name,
                                         f"size mismatch: checkpoint={param.shape}, model={model_state[name].shape}"))
            else:
                skipped_keys.append((name, "not found in current model"))

        # Load the filtered state dict
        self.load_state_dict(model_state, strict=False)

        # Log loading statistics
        logger.info("\nPretrained weight loading summary:")
        logger.info(f"Successfully loaded {len(loaded_keys)} layers:")
        for key in loaded_keys:
            logger.info(f"  - {key}")

        logger.info(f"\nSkipped {len(skipped_keys)} layers:")
        for key, reason in skipped_keys:
            logger.info(f"  - {key}: {reason}")

        # Log layer status for backbone
        logger.info("\nBackbone layer loading status:")
        total_backbone_params = 0
        loaded_backbone_params = 0

        for name, param in self.backbone.named_parameters():
            total_backbone_params += param.numel()
            if f"backbone.{name}" in loaded_keys:
                loaded_backbone_params += param.numel()

        loading_percentage = (loaded_backbone_params / total_backbone_params) * 100
        logger.info(f"Loaded {loading_percentage:.1f}% of backbone parameters")




class TripletPatchTSTClassifier(PatchTSTClassifier):
    def __init__(
            self,
            c_in: int,
            patch_len: int,
            stride: int,
            num_patch: int,
            n_classes: int = 2,
            d_model: int = 128,
            n_heads: int = 16,
            n_layers: int = 3,
            d_ff: int = 256,
            dropout: float = 0.1,
            head_dropout: float = 0.1,
            attn_dropout: float = 0.1,
            embedding_dim: int = 128,
            margin: float = 1.0,
            **kwargs
    ):
        super().__init__(
            c_in=c_in,
            patch_len=patch_len,
            stride=stride,
            num_patch=num_patch,
            n_classes=n_classes,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            head_dropout=head_dropout,
            attn_dropout=attn_dropout,
            **kwargs
        )

        # Calculate the correct input dimension for the embedding head
        self.patch_len = patch_len
        self.stride = stride
        self.num_patch = num_patch
        self.embedding_dim = embedding_dim
        self.margin = margin

        # Calculate feature dimension after backbone
        feature_dim = c_in * d_model* num_patch

        # Replace the original head with embedding + classification heads
        self.embedding_head = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(embedding_dim, n_classes)
        )

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with corrected dimensionality handling

        Args:
            x: Input tensor of shape [batch_size, features, seq_len]

        Returns:
            During training: (embeddings, logits)
            During inference: logits
        """
        batch_size, features, seq_len = x.shape

        # Process through backbone
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, 1, features, self.patch_len)
        x = self.backbone(x)  # [batch_size, c_in, d_model, num_patch]

        # Flatten all dimensions except batch
        x = x.reshape(batch_size, -1)  # [batch_size, c_in * d_model * num_patch]

        # Debug print

        # Get embeddings
        embeddings = self.embedding_head(x)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Get classifications
        logits = self.classification_head(embeddings)

        return embeddings, logits

    def train_model(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            test_loader: Optional[DataLoader] = None,
            epochs: int = 50,
            learning_rate: float = 0.001,
            weight_decay: float = 0.01,
            patience: int = 10,
            output_dir: Optional[Path] = None,
            device: Optional[torch.device] = None,
            class_weights: Optional[torch.Tensor] = None
    ) -> Dict:
        """Modified training loop with detailed logging"""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize losses
        triplet_criterion = TripletLoss(margin=self.margin)
        if class_weights is not None:
            classification_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            classification_criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=epochs // 3, T_mult=2
        )

        # Training history
        history = {
            'train_loss': [], 'train_acc': [], 'train_f1': [],
            'train_triplet_loss': [], 'train_classification_loss': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [],
            'learning_rates': []
        }

        # Early stopping setup
        best_val_loss = float('inf')
        best_epoch = 0
        best_model = None
        no_improve = 0

        self.logger.info(f"\nStarting training on {device}")
        self.logger.info(f"Total epochs: {epochs}")
        self.logger.info(f"Learning rate: {learning_rate}")
        self.logger.info(f"Weight decay: {weight_decay}")

        try:
            for epoch in range(epochs):
                epoch_start = time.time()

                # Training
                train_metrics = self._train_epoch_with_triplet(
                    train_loader,
                    triplet_criterion,
                    classification_criterion,
                    optimizer,
                    device,
                    epoch,
                    epochs
                )

                # Validation
                if val_loader is not None:
                    val_metrics = self._validate_with_triplet(
                        val_loader,
                        triplet_criterion,
                        classification_criterion,
                        device
                    )
                    # Early stopping check
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        best_epoch = epoch
                        best_model = self.state_dict()
                        no_improve = 0

                        if output_dir:
                            self._save_checkpoint(
                                output_dir / 'best_model.pth',
                                epoch,
                                optimizer,
                                val_metrics
                            )
                    else:
                        no_improve += 1

                # Update learning rate
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]

                # Update history
                history['train_loss'].append(train_metrics['loss'])
                history['train_acc'].append(train_metrics['acc'])
                history['train_f1'].append(train_metrics['f1'])
                history['train_triplet_loss'].append(train_metrics['triplet_loss'])
                history['train_classification_loss'].append(train_metrics['classification_loss'])
                history['learning_rates'].append(current_lr)

                if val_loader is not None:
                    history['val_loss'].append(val_metrics['loss'])
                    history['val_acc'].append(val_metrics['acc'])
                    history['val_f1'].append(val_metrics['f1'])

                # Log progress
                epoch_time = time.time() - epoch_start
                self._log_epoch_triplet(
                    epoch, epochs, train_metrics, val_metrics if val_loader else None,
                    current_lr, epoch_time
                )

                # Save current state
                if output_dir:
                    self._save_checkpoint(
                        output_dir / 'last_model.pth',
                        epoch,
                        optimizer,
                        val_metrics if val_loader else train_metrics
                    )
                    self._save_history(history, output_dir / 'history.json')

                # Early stopping
                if no_improve >= patience:
                    self.logger.info(f"\nEarly stopping triggered! No improvement for {patience} epochs")
                    break

        except KeyboardInterrupt:
            self.logger.info("\nTraining interrupted by user!")

        # Load best model if available
        if best_model is not None:
            self.load_state_dict(best_model)
            self.logger.info(f"\nLoaded best model from epoch {best_epoch}")

        return history


    def _train_epoch_with_triplet(
            self,
            train_loader: DataLoader,
            triplet_criterion: nn.Module,
            classification_criterion: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            epoch: int,
            total_epochs: int
    ) -> Dict:
        """Train for one epoch with triplet loss"""
        self.train()
        metrics = {
            'loss': 0.0,
            'triplet_loss': 0.0,
            'classification_loss': 0.0,
            'correct': 0,
            'total': 0,
            'predictions': [],
            'targets': [],
            'embeddings': []
        }

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epochs}',
                    leave=False, dynamic_ncols=True)

        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move data to device
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_size = inputs.size(0)

            # Forward pass
            optimizer.zero_grad(set_to_none=True)
            embeddings, logits = self(inputs)

            # Calculate losses
            triplet_loss = triplet_criterion(embeddings, targets)
            classification_loss = classification_criterion(logits, targets)
            loss = 0.5 * triplet_loss + 0.5 * classification_loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()

            # Update metrics
            metrics['loss'] += loss.item() * batch_size
            metrics['triplet_loss'] += triplet_loss.item() * batch_size
            metrics['classification_loss'] += classification_loss.item() * batch_size

            _, predicted = logits.max(1)
            metrics['total'] += batch_size
            metrics['correct'] += predicted.eq(targets).sum().item()

            # Store predictions and targets for confusion matrix
            metrics['predictions'].extend(predicted.cpu().numpy())
            metrics['targets'].extend(targets.cpu().numpy())
            metrics['embeddings'].extend(embeddings.detach().cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * metrics['correct'] / metrics['total']:.2f}%",
            })

        return self._calculate_metrics(metrics, len(train_loader))

    def _calculate_metrics(self, metrics: Dict, num_batches: int, phase: str = 'Train') -> Dict:
        """Calculate metrics for the epoch"""
        if not metrics['targets']:  # Check if we have any targets
            return {}

        calculated = {
            'loss': metrics['loss'] / metrics['total'],
            'triplet_loss': metrics.get('triplet_loss', 0) / metrics['total'],
            'classification_loss': metrics.get('classification_loss', 0) / metrics['total'],
            'acc': 100. * metrics['correct'] / metrics['total'],
            'predictions': np.array(metrics['predictions']),
            'targets': np.array(metrics['targets'])
        }

        # Calculate additional metrics
        calculated['f1'] = f1_score(
            calculated['targets'],
            calculated['predictions'],
            average='weighted'
        )
        calculated['precision'] = precision_score(
            calculated['targets'],
            calculated['predictions'],
            average='weighted'
        )
        calculated['recall'] = recall_score(
            calculated['targets'],
            calculated['predictions'],
            average='weighted'
        )

        return calculated

    def _log_epoch_triplet(
            self,
            epoch: int,
            total_epochs: int,
            train_metrics: Dict,
            val_metrics: Optional[Dict],
            lr: float,
            epoch_time: float
    ):
        """Log epoch results for triplet training"""
        self.logger.info(
            f"\nEpoch {epoch + 1}/{total_epochs} - {epoch_time:.2f}s - LR: {lr:.6f}"
        )

        # Train metrics
        if train_metrics:
            self.logger.info(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Triplet Loss: {train_metrics['triplet_loss']:.4f}, "
                f"Class Loss: {train_metrics['classification_loss']:.4f}, "
                f"Acc: {train_metrics['acc']:.2f}%, "
                f"F1: {train_metrics['f1']:.4f}"
            )

            if 'targets' in train_metrics and 'predictions' in train_metrics:
                self.logger.info("\nTraining Confusion Matrix:")
                self.plot_confusion_matrix(
                    train_metrics['targets'],
                    train_metrics['predictions']
                )

        # Validation metrics
        if val_metrics:
            self.logger.info(
                f"Val - Loss: {val_metrics['loss']:.4f}, "
                f"Triplet Loss: {val_metrics['triplet_loss']:.4f}, "
                f"Class Loss: {val_metrics['classification_loss']:.4f}, "
                f"Acc: {val_metrics['acc']:.2f}%, "
                f"F1: {val_metrics['f1']:.4f}"
            )

            if 'targets' in val_metrics and 'predictions' in val_metrics:
                self.logger.info("\nValidation Confusion Matrix:")
                self.plot_confusion_matrix(
                    val_metrics['targets'],
                    val_metrics['predictions']
                )

    def plot_confusion_matrix(self, targets: np.ndarray, predictions: np.ndarray):
        """Plot confusion matrix"""
        cm = confusion_matrix(targets, predictions)

        # Convert counts to percentages
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        # Print the confusion matrix
        self.logger.info("\nConfusion Matrix (%):")
        for i in range(cm.shape[0]):
            row = " ".join(f"{val:6.1f}" for val in cm_percentage[i])
            self.logger.info(f"Class {i}: [{row}]")

        # Print additional metrics per class
        precision = precision_score(targets, predictions, average=None)
        recall = recall_score(targets, predictions, average=None)
        f1 = f1_score(targets, predictions, average=None)

        self.logger.info("\nPer-class metrics:")
        for i in range(len(precision)):
            self.logger.info(
                f"Class {i}: Precision={precision[i]:.3f}, "
                f"Recall={recall[i]:.3f}, F1={f1[i]:.3f}"
            )
    def _validate_with_triplet(
            self,
            dataloader: DataLoader,
            triplet_criterion: nn.Module,
            classification_criterion: nn.Module,
            device: torch.device
    ) -> Dict:
        """
        Validate the model with both triplet and classification losses.

        Args:
            dataloader: Validation data loader
            triplet_criterion: Triplet loss criterion
            classification_criterion: Classification loss criterion
            device: Device to run validation on

        Returns:
            Dictionary containing validation metrics
        """
        self.eval()
        metrics = {
            'loss': 0.0,
            'triplet_loss': 0.0,
            'classification_loss': 0.0,
            'correct': 0,
            'total': 0,
            'predictions': [],
            'targets': []
        }

        with torch.no_grad():
            for inputs, targets in dataloader:
                # Move data to device
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                batch_size = inputs.size(0)

                # Forward pass
                embeddings, logits = self(inputs)

                # Calculate losses
                triplet_loss = triplet_criterion(embeddings, targets)
                classification_loss = classification_criterion(logits, targets)

                # Combined loss
                loss = 0.5 * triplet_loss + 0.5 * classification_loss

                # Update metrics
                metrics['loss'] += loss.item() * batch_size
                metrics['triplet_loss'] += triplet_loss.item() * batch_size
                metrics['classification_loss'] += classification_loss.item() * batch_size

                # Calculate accuracy
                _, predicted = logits.max(1)
                metrics['total'] += batch_size
                metrics['correct'] += predicted.eq(targets).sum().item()

                # Store predictions and targets for later analysis
                metrics['predictions'].extend(predicted.cpu().numpy())
                metrics['targets'].extend(targets.cpu().numpy())

        return self._calculate_metrics(metrics, len(dataloader), 'Validation')