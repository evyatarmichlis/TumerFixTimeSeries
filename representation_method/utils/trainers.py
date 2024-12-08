import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sympy.abc import alpha
from tqdm import tqdm
import matplotlib

from plots_helper import labels
from representation_method.utils.losses import ContrastiveAutoencoderLoss

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


class BaseTrainer:
    """Base trainer class with common functionality"""

    def __init__(self, model, criterion, optimizer, scheduler, device, save_path,early_stopping_patience=5):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path
        self.early_stopping_patience = early_stopping_patience

        # Create save directory if it doesn't exist
        if save_path:
            os.makedirs(save_path, exist_ok=True)

    def save_checkpoint(self, path, epoch=None, best=False, model_type="model"):
        """
        Save complete training checkpoint including optimizer and scheduler states
        for resuming training later
        """
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch
        }
        filename = f'checkpoint_{model_type}_epoch_{epoch}.pth'
        if best:
            filename = f'best_{model_type}_checkpoint.pth'
        torch.save(save_dict, os.path.join(path, filename))

    def save_model(self, path, epoch=None, best=False, model_type="model"):
        """
        Save only the model state dict for inference
        """
        filename = f'model_{model_type}_epoch_{epoch}.pth'
        if best:
            filename = f'best_{model_type}_model.pth'
        torch.save(self.model.state_dict(), os.path.join(path, filename))

    def load_checkpoint(self, checkpoint_path):
        """
        Load complete checkpoint for resuming training
        """
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        return epoch

    def load_model(self, model_path):
        """
        Load just the model state dict for inference
        """
        state_dict = torch.load(model_path)
        # Handle both cases where the file contains just the state dict
        # or a full checkpoint
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        self.model.load_state_dict(state_dict)



class AutoencoderTrainer(BaseTrainer):
    """Trainer class for autoencoder"""

    def __init__(self, model, criterion, optimizer, scheduler, device,
                 mask_probability=0.1, save_path=None,early_stopping_patience=5):
        super().__init__(model, criterion, optimizer, scheduler, device, save_path,early_stopping_patience)
        self.mask_probability = mask_probability
        self.best_val_loss = float('inf')
        self.early_stopping_patience = early_stopping_patience

    def apply_mask(self, inputs):
        """Apply random masking to inputs"""
        if self.mask_probability > 0:
            mask = torch.bernoulli(
                torch.full(inputs.shape, 1 - self.mask_probability)).to(self.device)
            return inputs * mask
        return inputs

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        for inputs, _ in train_loader:
            inputs = inputs.to(self.device)
            inputs = self.apply_mask(inputs)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, inputs)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)


    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(self.device)
                inputs = self.apply_mask(inputs)
                output = self.model(inputs)
                loss = self.criterion(output, inputs)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs):
        """
        Main training loop with improved checkpointing and monitoring
        """
        train_losses = []
        val_losses = []
        patience_counter = 0
        start_time = time.time()

        # Initialize tracking variables
        self.best_val_loss = float('inf')
        best_epoch = -1

        print(f"Starting training for {epochs} epochs...")

        try:
            for epoch in range(epochs):
                epoch_start_time = time.time()

                # Training phase
                self.model.train()
                train_loss = self.train_epoch(train_loader)
                train_losses.append(train_loss)

                # Validation phase
                self.model.eval()
                val_loss = self.validate(val_loader)
                val_losses.append(val_loss)

                # Learning rate scheduling
                if self.scheduler:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    self.scheduler.step(val_loss)
                    new_lr = self.optimizer.param_groups[0]['lr']
                    if old_lr != new_lr:
                        print(f"Learning rate adjusted from {old_lr:.2e} to {new_lr:.2e}")

                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time

                # Print progress
                print(f"\nEpoch {epoch + 1}/{epochs} - Time: {epoch_time:.2f}s")
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                # Save periodic checkpoint (every 10 epochs)
                if self.save_path and (epoch + 1) % 10 == 0:
                    self.save_checkpoint(
                        self.save_path,
                        epoch=epoch,
                        model_type="autoencoder"
                    )

                # Check for best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    print(f"New best model! Validation loss improved to {val_loss:.4f}")

                    if self.save_path:
                        # Save both checkpoint and model state
                        self.save_checkpoint(
                            self.save_path,
                            epoch=epoch,
                            best=True,
                            model_type="autoencoder"
                        )
                        self.save_model(
                            self.save_path,
                            epoch=epoch,
                            best=True,
                            model_type="autoencoder"
                        )
                else:
                    patience_counter += 1
                    print(f"Validation loss did not improve. Best: {self.best_val_loss:.4f} "
                          f"at epoch {best_epoch + 1}. Patience: {patience_counter}/{self.early_stopping_patience}")

                    if patience_counter >= self.early_stopping_patience:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                        print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {best_epoch + 1}")
                        break

                print("-" * 80)  # Separator between epochs

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        finally:
            # Training summary
            total_time = time.time() - start_time
            print("\nTraining completed!")
            print(f"Total training time: {total_time:.2f}s")
            print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {best_epoch + 1}")

            # Save final results_old
            if self.save_path:
                # Save training curves
                self._plot_training_curves(train_losses, val_losses)

                # Save training history
                history = {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'best_epoch': best_epoch,
                    'best_val_loss': self.best_val_loss,
                    'total_epochs': epoch + 1,
                    'total_time': total_time
                }

                with open(os.path.join(self.save_path, 'training_history.json'), 'w') as f:
                    json.dump(history, f)

    def _plot_training_curves(self, train_losses, val_losses, title="Loss", suffix=""):
        """Plot and save training curves with custom title and suffix"""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training')
        plt.plot(val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.title(f'Training and Validation {title}')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, f'training_curves{suffix}.png'))
        plt.close()




class CombinedModelTrainer(BaseTrainer):

    def __init__(self, model, criterion, optimizer, scheduler, device, save_path, early_stopping_patience=5):
        super().__init__(model, criterion, optimizer, scheduler, device, save_path, early_stopping_patience)
        self.best_val_loss = float('inf')

    def train_epoch(self, train_loader, window_weights=None):
        """Train for one epoch with progress bar"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            weights = torch.tensor(window_weights[i], dtype=torch.float32).to(
                self.device) if window_weights is not None else None
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            if weights is not None:
                loss = self.criterion(outputs, labels, weights)
            else:
                loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            batch_correct = predicted.eq(labels).sum().item()
            batch_total = labels.size(0)
            total += batch_total
            correct += batch_correct



        return total_loss / len(train_loader), correct / total


    def validate(self, val_loader, window_weights=None):
        """Validate with progress bar"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                weights = torch.tensor(window_weights[i], dtype=torch.float32).to(
                    self.device) if window_weights is not None else None
                outputs = self.model(inputs)

                if weights is not None:
                    loss = self.criterion(outputs, labels, weights)
                else:
                    loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(labels).sum().item()
                batch_total = labels.size(0)
                total += batch_total
                correct += batch_correct


        return total_loss / len(val_loader), correct / total

    def find_optimal_threshold(self, val_loader, n_thresholds=100, max_fp_tp_ratio=5.0, min_recall=0.3):
        """
        Find the optimal threshold based on validation set metrics, controlling FP/TP ratio.

        Args:
            val_loader: Validation data loader
            n_thresholds: Number of threshold values to try
            max_fp_tp_ratio: Maximum allowed ratio of FP/TP
            min_recall: Minimum required recall to ensure we're not being too conservative

        Returns:
            float: Optimal threshold value
            dict: Detailed metrics for the optimal threshold
            list: All results_old for analysis
        """
        self.model.eval()
        all_probs = []
        all_labels = []

        # Collect all predictions and labels
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Collecting predictions'):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.sigmoid(outputs[:, 0]).cpu().numpy()
                all_probs.extend(probabilities)
                all_labels.extend(labels.numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Try different thresholds
        thresholds = np.linspace(0.5, 0.95, n_thresholds)
        results = []

        for threshold in tqdm(thresholds, desc='Finding optimal threshold'):
            predictions = (all_probs >= threshold).astype(int)

            # Calculate metrics
            tp = np.sum((predictions == 1) & (all_labels == 1))
            fp = np.sum((predictions == 1) & (all_labels == 0))
            fn = np.sum((predictions == 0) & (all_labels == 1))
            tn = np.sum((predictions == 0) & (all_labels == 0))

            # Calculate various metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fp_tp_ratio = fp / tp if tp > 0 else float('inf')
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            results.append({
                'threshold': threshold,
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'fp_tp_ratio': fp_tp_ratio,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            })

        # Filter results_old based on criteria
        valid_results = [r for r in results if r['fp_tp_ratio'] <= max_fp_tp_ratio
                         and r['recall'] >= min_recall]

        if not valid_results:
            print("No threshold satisfies the criteria. Consider relaxing constraints.")
            # Return the result with the lowest FP/TP ratio that meets minimum recall
            valid_results = [r for r in results if r['recall'] >= min_recall]
            if valid_results:
                best_result = min(valid_results, key=lambda x: x['fp_tp_ratio'])
            else:
                best_result = max(results, key=lambda x: x['f1'])
        else:
            # Among valid results_old, choose the one with highest F1 score
            best_result = max(valid_results, key=lambda x: x['f1'])

        # Create visualization of metrics
        thresholds = [r['threshold'] for r in results]
        fp_tp_ratios = [r['fp_tp_ratio'] if r['fp_tp_ratio'] != float('inf') else max_fp_tp_ratio * 2
                        for r in results]
        recalls = [r['recall'] for r in results]
        precisions = [r['precision'] for r in results]
        f1_scores = [r['f1'] for r in results]

        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, fp_tp_ratios, label='FP/TP Ratio')
        plt.plot(thresholds, recalls, label='Recall')
        plt.plot(thresholds, precisions, label='Precision')
        plt.plot(thresholds, f1_scores, label='F1 Score')
        plt.axvline(x=best_result['threshold'], color='r', linestyle='--',
                    label=f"Selected Threshold: {best_result['threshold']:.3f}")
        plt.axhline(y=max_fp_tp_ratio, color='g', linestyle='--',
                    label=f'Max FP/TP Ratio: {max_fp_tp_ratio}')
        plt.axhline(y=min_recall, color='y', linestyle='--',
                    label=f'Min Recall: {min_recall}')
        plt.xlabel('Threshold')
        plt.ylabel('Metric Value')
        plt.title('Metrics vs Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.save_path, 'optimal_threshold.png'))

        return best_result['threshold'], best_result, results

    def evaluate(self, test_loader, threshold=None):
        """Evaluate with optimal or provided threshold"""
        if threshold is None:
            threshold_path = os.path.join(self.save_path, 'best_threshold.json')
            if os.path.exists(threshold_path):
                with open(threshold_path, 'r') as f:
                    threshold = json.load(f)['threshold']
            else:
                threshold = 0.5  
                print("Warning: No optimal threshold found, using default 0.5")

        self.model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Evaluating'):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.sigmoid(outputs[:, 0]).cpu().numpy()
                all_probs.extend(probabilities)
                all_labels.extend(labels.numpy())

        predictions = (np.array(all_probs) >= threshold).astype(int)
        labels = np.array(all_labels)

        # Calculate metrics
        report = classification_report(labels, predictions)
        cm = confusion_matrix(labels, predictions)

        if self.save_path:
            with open(os.path.join(self.save_path, 'evaluation_results.txt'), 'w') as f:
                f.write(f"Using threshold: {threshold}\n\n")
                f.write("Classification Report:\n")
                f.write(report)
                f.write("\nConfusion Matrix:\n")
                f.write("   Predicted 0  Predicted 1\n")
                f.write(f"Actual 0   {cm[0, 0]:<10} {cm[0, 1]:<10}\n")
                f.write(f"Actual 1   {cm[1, 0]:<10} {cm[1, 1]:<10}\n")

        return report, cm, threshold

    def train(self, train_loader, val_loader, epochs, window_weights_train=None, window_weights_val=None):
        """Main training loop with epoch progress bar"""
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        patience_counter = 0

        # Main epoch progress bar
        epoch_pbar = tqdm(range(epochs), desc='Training Progress',leave=False)
        for epoch in epoch_pbar:
            train_loss, train_acc = self.train_epoch(train_loader, window_weights_train)
            val_loss, val_acc = self.validate(val_loader, window_weights_val)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if self.scheduler:
                self.scheduler.step(val_loss)

            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'train_acc': f'{100. * train_acc:.2f}%',
                'val_loss': f'{val_loss:.4f}',
                'val_acc': f'{100. * val_acc:.2f}%'
            })

            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                if self.save_path:
                    self.save_model(self.save_path, epoch, best=True,model_type='classifier')
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break


        if self.save_path:
            self._plot_training_curves(train_losses, val_losses, train_accs, val_accs,'classifer')

        print("\nFinding optimal threshold...")
        best_threshold, best_f1, _ = self.find_optimal_threshold(val_loader)

        return best_threshold
    def _plot_training_curves(self, train_losses, val_losses, train_accs, val_accs,name ='classifer'):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()

        # Accuracy plot
        ax2.plot(train_accs, label='Training Accuracy')
        ax2.plot(val_accs, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'training_curves_{name}.png'))
        plt.close()


class ContrastiveAutoencoderTrainer(AutoencoderTrainer):
    """Trainer class for contrastive autoencoder"""

    def __init__(self, model, optimizer, loss_function, device,
                 scheduler=None, mask_probability=0.1, save_path=None,
                 early_stopping_patience=5):
        # Pass loss_function as criterion to match parent's signature
        super().__init__(
            model=model,
            criterion=loss_function,  # Use loss_function as criterion
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            mask_probability=mask_probability,
            save_path=save_path,
            early_stopping_patience=early_stopping_patience
        )

    def train_epoch(self, train_loader):
        """Train for one epoch with contrastive loss"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_contrast_loss = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            inputs = self.apply_mask(inputs)

            self.optimizer.zero_grad()
            encoder_output = self.model.encoder(inputs)  # This returns (outputs, hidden)

            reconstructed = self.model(inputs)

            loss, recon_loss, contrast_loss = self.criterion.calculate_loss(
                reconstructed,
                inputs,
                encoder_output,  # Pass the full encoder output (outputs, hidden)
                labels
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_contrast_loss += contrast_loss.item()

        avg_loss = total_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_contrast_loss = total_contrast_loss / len(train_loader)

        return avg_loss, avg_recon_loss, avg_contrast_loss

    def validate(self, val_loader):
        """Validate with contrastive loss"""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_contrast_loss = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs = self.apply_mask(inputs)

                # Get encoder outputs and reconstruction
                encoder_output = self.model.encoder(inputs)  # This returns (outputs, hidden)

                reconstructed = self.model(inputs)

                loss, recon_loss, contrast_loss = self.criterion.calculate_loss(
                    reconstructed,
                    inputs,
                    encoder_output,  # Pass the full encoder output (outputs, hidden)
                    labels
                )

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_contrast_loss += contrast_loss.item()

        avg_loss = total_loss / len(val_loader)
        avg_recon_loss = total_recon_loss / len(val_loader)
        avg_contrast_loss = total_contrast_loss / len(val_loader)

        return avg_loss, avg_recon_loss, avg_contrast_loss

    def train(self, train_loader, val_loader, epochs):
        """
        Main training loop with improved checkpointing and monitoring
        """
        train_losses = []
        val_losses = []
        train_recon_losses = []
        val_recon_losses = []
        train_contrast_losses = []
        val_contrast_losses = []
        patience_counter = 0
        start_time = time.time()

        # Initialize tracking variables
        self.best_val_loss = float('inf')
        best_epoch = -1
        last_epoch = 0

        print(f"Starting training for {epochs} epochs...")

        try:
            for epoch in range(epochs):
                last_epoch = epoch
                epoch_start_time = time.time()

                # Training phase
                self.model.train()
                train_loss, train_recon, train_contrast = self.train_epoch(train_loader)

                # Debug print

                train_losses.append(float(train_loss))
                train_recon_losses.append(float(train_recon))
                train_contrast_losses.append(float(train_contrast))

                # Validation phase
                self.model.eval()
                val_loss, val_recon, val_contrast = self.validate(val_loader)

                # Debug print

                val_losses.append(float(val_loss))
                val_recon_losses.append(float(val_recon))
                val_contrast_losses.append(float(val_contrast))

                # Make sure val_loss is a float for comparison
                val_loss_float = float(val_loss)

                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time

                # Print progress
                print(f"\nEpoch {epoch + 1}/{epochs} - Time: {epoch_time:.2f}s")
                print(f"Train - Total: {train_loss:.4f}, Recon: {train_recon:.4f}, Contrast: {train_contrast:.4f}")
                print(f"Val - Total: {val_loss:.4f}, Recon: {val_recon:.4f}, Contrast: {val_contrast:.4f}")

                # Learning rate scheduling
                if self.scheduler is not None:
                    self.scheduler.step(val_loss_float)

                # Check for best model
                if val_loss_float < self.best_val_loss:
                    self.best_val_loss = val_loss_float
                    best_epoch = epoch
                    patience_counter = 0

                    if self.save_path:
                        self.save_checkpoint(
                            self.save_path,
                            epoch=epoch,
                            best=True,
                            model_type="autoencoder"
                        )
                else:
                    patience_counter += 1
                    print(f"Validation loss did not improve. Best: {self.best_val_loss:.4f} "
                          f"at epoch {best_epoch + 1}. Patience: {patience_counter}/{self.early_stopping_patience}")

                    if patience_counter >= self.early_stopping_patience:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                        break

                # Save periodic checkpoint
                if self.save_path and (epoch + 1) % 10 == 0:
                    self.save_checkpoint(
                        self.save_path,
                        epoch=epoch,
                        model_type="autoencoder"
                    )

                print("-" * 80)

        except Exception as e:
            print(f"\nError during training: {str(e)}")
            import traceback
            traceback.print_exc()

        finally:
            total_time = time.time() - start_time
            print("\nTraining completed!")
            print(f"Total time: {total_time:.2f}s")
            print(f"Epochs completed: {last_epoch + 1}")
            print(f"Best val loss: {self.best_val_loss:.4f} at epoch {best_epoch + 1}")

            if self.save_path and len(train_losses) > 0:
                # Save plots and history
                self._plot_training_curves(train_losses, val_losses, "Total Loss")
                self._plot_training_curves(train_recon_losses, val_recon_losses, "Reconstruction Loss", suffix="_recon")
                self._plot_training_curves(train_contrast_losses, val_contrast_losses, "Contrastive Loss",
                                           suffix="_contrast")

                history = {
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'train_recon_losses': train_recon_losses,
                    'val_recon_losses': val_recon_losses,
                    'train_contrast_losses': train_contrast_losses,
                    'val_contrast_losses': val_contrast_losses,
                    'best_epoch': best_epoch,
                    'best_val_loss': float(self.best_val_loss),
                    'total_epochs': last_epoch + 1,
                    'total_time': total_time
                }

                with open(os.path.join(self.save_path, 'training_history.json'), 'w') as f:
                    json.dump(history, f)

            return train_losses, val_losses

import torch.nn.functional as F

class VAETrainer(BaseTrainer):
    """Trainer class specifically for VAE"""

    def __init__(self, model,criterion, optimizer, scheduler, device,l2_alpha =0.1, beta=0.1,margin=1,triplet_weight=1,distance_metric='L2',loss_type='normal',
                 save_path=None, early_stopping_patience=10):
        super().__init__(model,criterion, optimizer, scheduler, device, save_path, early_stopping_patience)
        self.l2_alpha = l2_alpha
        self.beta = beta  # Weight for KL divergence loss
        self.margin = margin
        self.best_val_loss = float('inf')
        self.triplet_weight = triplet_weight
        self.distance_metric = distance_metric
        self.loss_type = loss_type

    def compute_loss(self, x, recon_x, mu, logvar):
        """Compute VAE loss: reconstruction + KL divergence"""
        # Reconstruction loss (MSE for time series)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss


    def choose_triplet_loss(self, x, recon_x, mu, logvar, labels):
        if self.loss_type == 'normal':
            return self.compute_triplet_loss(x, recon_x, mu, logvar, labels)
        elif self.loss_type == 'dual':
            return self.compute_dual_triplet_loss(x, recon_x, mu, logvar, labels)
        else:
            return -1

    def compute_triplet_loss(self, x, recon_x, mu, logvar, labels):
        """Compute VAE loss: reconstruction + KL divergence + triplet loss"""
        # Original losses
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Create minority mask as tensor
        minority_mask = (labels == 1)
        triplet_loss = torch.tensor(0.0, device=mu.device)

        if torch.any(minority_mask):  # Use torch.any() instead of .any()
            # Normalize embeddings
            embeddings = F.normalize(mu, p=2, dim=1)
            # Compute pairwise distances
            if self.distance_metric == "L2":
                dist_matrix = torch.cdist(embeddings, embeddings)
            elif self.distance_metric == "cosine":
                cos_sim_matrix = torch.mm(embeddings, embeddings.t())
                # Convert to distances (1 - similarity)
                dist_matrix = 1 - cos_sim_matrix
            # Get hardest positive and negative for minority class
            minority_indices = torch.where(minority_mask)[0]
            for anchor_idx in minority_indices:
                pos_mask = (labels == labels[anchor_idx]) & (
                            torch.arange(len(labels), device=labels.device) != anchor_idx)
                neg_mask = labels != labels[anchor_idx]
                if torch.any(pos_mask) and torch.any(neg_mask):
                    # Get hardest positive and negative
                    pos_dist = dist_matrix[anchor_idx][pos_mask].mean()
                    neg_dist = dist_matrix[anchor_idx][neg_mask].min()
                    triplet_loss += torch.relu(pos_dist - neg_dist +  self.margin)
        total_loss = self.l2_alpha *recon_loss + self.beta * kl_loss + self.triplet_weight * triplet_loss
        return total_loss, recon_loss, kl_loss , triplet_loss

    def compute_adaptive_margin(self,dist_matrix, labels):
        intra_class_dist = dist_matrix[labels.unsqueeze(0) == labels.unsqueeze(1)].mean()
        inter_class_dist = dist_matrix[labels.unsqueeze(0) != labels.unsqueeze(1)].mean()
        return (inter_class_dist - intra_class_dist) * 0.5


    def compute_dual_triplet_loss(self, x, recon_x, mu, logvar, labels):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        embeddings = F.normalize(mu, p=2, dim=1)
        if self.distance_metric == "L2":
            dist_matrix = torch.cdist(embeddings, embeddings)
        elif self.distance_metric == "cosine":
            cos_sim_matrix = torch.mm(embeddings, embeddings.t())
            dist_matrix = 1 - cos_sim_matrix

        triplet_loss = torch.tensor(0.0, device=mu.device)

        # Compute adaptive margin
        margin = self.margin

        for class_label in [0, 1]:
            class_indices = torch.where(labels == class_label)[0]
            for anchor_idx in class_indices:
                pos_mask = (labels == labels[anchor_idx]) & (
                            torch.arange(len(labels), device=labels.device) != anchor_idx)
                neg_mask = labels != labels[anchor_idx]

                if torch.any(pos_mask) and torch.any(neg_mask):
                    pos_dist = dist_matrix[anchor_idx][pos_mask].mean()
                    neg_dist = dist_matrix[anchor_idx][neg_mask].min()
                    triplet_loss += torch.relu(pos_dist - neg_dist + margin)

        total_loss = recon_loss + self.beta * kl_loss + self.triplet_weight * triplet_loss
        return total_loss, recon_loss, kl_loss, triplet_loss

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_triplet_loss = 0
        for inputs,labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass
            recon_batch, mu, logvar = self.model(inputs)

            # Compute loss
            loss, recon_loss, kl_loss,triplet_loss = self.choose_triplet_loss(inputs, recon_batch, mu, logvar,labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            total_triplet_loss += triplet_loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon = total_recon_loss / len(train_loader.dataset)
        avg_kl = total_kl_loss / len(train_loader.dataset)
        avg_trip= total_triplet_loss / len(train_loader.dataset)

        return avg_loss, avg_recon, avg_kl,avg_trip

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0
        total_triplet_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                recon_batch, mu, logvar = self.model(inputs)

                # Compute loss
                loss, recon_loss, kl_loss,triplet_loss  = self.choose_triplet_loss(inputs, recon_batch, mu, logvar,labels)

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
                total_triplet_loss += triplet_loss.item()

        avg_loss = total_loss / len(val_loader.dataset)
        avg_recon = total_recon_loss / len(val_loader.dataset)
        avg_kl = total_kl_loss / len(val_loader.dataset)
        avg_trip= total_triplet_loss / len(val_loader.dataset)

        return avg_loss, avg_recon, avg_kl,avg_trip

    def train(self, train_loader, val_loader, epochs):
        """Main training loop"""
        train_losses = []
        val_losses = []
        train_recon_losses = []
        val_recon_losses = []
        train_kl_losses = []
        val_kl_losses = []
        train_triplet_losses = []
        val_triplet_losses = []
        patience_counter = 0

        print(f"Starting training for {epochs} epochs...")

        try:
            for epoch in range(epochs):
                # Training phase
                train_loss, train_recon, train_kl,train_triplet = self.train_epoch(train_loader)
                train_losses.append(train_loss)
                train_recon_losses.append(train_recon)
                train_kl_losses.append(train_kl)
                train_triplet_losses.append(train_triplet)
                # Validation phase
                val_loss, val_recon, val_kl,val_triplet = self.validate(val_loader)
                val_losses.append(val_loss)
                val_recon_losses.append(val_recon)
                val_kl_losses.append(val_kl)
                val_triplet_losses.append(val_triplet)
                # Print progress
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print(f"Train - Total: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}, Triplet:{train_triplet:.4f}")
                print(f"Val - Total: {val_loss:.4f}, Recon: {val_recon:.4f}, KL: {val_kl:.4f}, Triplet:{val_triplet:.4f}")

                # Learning rate scheduling
                if self.scheduler:
                    self.scheduler.step(val_loss)

                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    patience_counter = 0
                    if self.save_path:
                        self.save_model(self.save_path, epoch, best=True)
                else:
                    patience_counter += 1
                    print(f"Validation loss did not improve. Best: {self.best_val_loss:.4f}")
                    if patience_counter >= self.early_stopping_patience:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                        break

                print("-" * 80)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        finally:
            if self.save_path:
                # Plot and save training curves
                self._plot_training_curves(
                    train_losses, val_losses,
                    train_recon_losses, val_recon_losses,
                    train_kl_losses, val_kl_losses
                )

        return train_losses, val_losses

    def _plot_training_curves(self, train_losses, val_losses,
                              train_recon_losses, val_recon_losses,
                              train_kl_losses, val_kl_losses):
        """Plot and save training curves"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Total loss
        ax1.plot(train_losses, label='Train')
        ax1.plot(val_losses, label='Val')
        ax1.set_title('Total Loss')
        ax1.legend()

        # Reconstruction loss
        ax2.plot(train_recon_losses, label='Train')
        ax2.plot(val_recon_losses, label='Val')
        ax2.set_title('Reconstruction Loss')
        ax2.legend()

        # KL loss
        ax3.plot(train_kl_losses, label='Train')
        ax3.plot(val_kl_losses, label='Val')
        ax3.set_title('KL Divergence')
        ax3.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, 'training_curves.png'))
        plt.close()