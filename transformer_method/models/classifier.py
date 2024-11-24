import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging
from PatchTST_self_supervised.src.models.patchTST import  PatchTSTEncoder, ClassificationHead


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
            dropout: float = 0.1,
            head_dropout: float = 0.1,
            attn_dropout: float = 0.1,
            act: str = "gelu",
            shared_embedding: bool = True,
            norm: str = 'BatchNorm',
            pe: str = 'zeros',
            learn_pe: bool = True,
            verbose: bool = False,
            **kwargs
    ):
        super().__init__()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape [batch_size, num_patch, n_vars, patch_len]

        Returns:
            Tensor of shape [batch_size, n_classes]
        """
        # Backbone forward pass
        x = self.backbone(x)  # [bs x nvars x d_model x num_patch]

        # Classification head
        x = self.head(x)  # [bs x n_classes]

        return x

    def load_pretrained(self, state_dict: Dict[str, Any], strict: bool = False):
        """
        Load pretrained weights with support for partial loading.

        Args:
            state_dict: Dictionary containing pretrained weights
            strict: Whether to strictly enforce that the keys match
        """
        # Filter out head parameters if they don't match
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and 'head' not in k}

        # Load filtered state dict
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)

        self.logger.info(f"Loaded pretrained weights for backbone")
        if not strict:
            self.logger.info("Classification head initialized randomly")

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
            output_dir: Optional[str] = None,
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

        # Training setup
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()

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
            self._log_metrics(test_metrics)
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
        """Train for one epoch"""
        self.train()
        metrics = {
            'loss': 0,
            'correct': 0,
            'total': 0,
            'predictions': [],
            'targets': []
        }

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{total_epochs}',
                    leave=False, dynamic_ncols=True)

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = self(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()

            metrics['loss'] += loss.item()
            _, predicted = outputs.max(1)
            metrics['total'] += targets.size(0)
            metrics['correct'] += predicted.eq(targets).sum().item()

            metrics['predictions'].extend(predicted.cpu().numpy())
            metrics['targets'].extend(targets.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100. * metrics['correct'] / metrics['total']:.2f}%"
            })

        return self._calculate_metrics(metrics, len(train_loader))

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

        return self._calculate_metrics(metrics, len(dataloader))

    def _calculate_metrics(self, metrics: Dict, num_batches: int) -> Dict:
        """Calculate classification metrics"""
        predictions = np.array(metrics['predictions'])
        targets = np.array(metrics['targets'])

        return {
            'loss': metrics['loss'] / num_batches,
            'acc': 100. * metrics['correct'] / metrics['total'],
            'precision': precision_score(targets, predictions, average='weighted', zero_division=0),
            'recall': recall_score(targets, predictions, average='weighted', zero_division=0),
            'f1': f1_score(targets, predictions, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(targets, predictions)
        }

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
        )
        if val_metrics:
            self.logger.info(
                f"Val - Loss: {val_metrics['loss']:.4f}, "
                f"Acc: {val_metrics['acc']:.2f}%, "
                f"F1: {val_metrics['f1']:.4f}"
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


