import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.manifold import TSNE
from sympy.abc import alpha
import matplotlib

from representation_method.resource_monitor import ResourceMonitor
from representation_method.utils.losses import ContrastiveAutoencoderLoss
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support
)
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


            # loss = self.criterion(outputs, labels,weights)
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


                # loss = self.criterion(outputs, labels, weights)
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
        monitor = ResourceMonitor(log_dir="resource_logs")
        # Main epoch progress bar
        monitor.start()
        best_val_loss = float('inf')

        epoch_pbar = tqdm(range(epochs), desc='Training Progress')
        for epoch in epoch_pbar:
            train_loss, train_acc = self.train_epoch(train_loader, window_weights_train)
            val_loss, val_acc = self.validate(val_loader, window_weights_val)
            best_val_loss = min(best_val_loss, val_loss)

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if self.scheduler:
                self.scheduler.step(val_loss)



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

            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'train_acc': f'{100. * train_acc:.2f}%',
                'val_loss': f'{val_loss:.4f}',
                'val_acc': f'{100. * val_acc:.2f}%',
                'best_val_loss': f'{best_val_loss:.4f}',
                'patience': patience_counter
            })



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
        monitor = ResourceMonitor(log_dir=os.path.join(self.save_path, 'resource_logs'))
        try:
            monitor.start()
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
                if epoch % 2 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"Error during training: {str(e)}")
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


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import logging


class VAEClassifierTrainer:
    def __init__(self, model, device, focal_loss_gamma=2.0):
        # [Previous init code remains the same]
        self.model = model
        self.device = device
        self.focal_loss_gamma = focal_loss_gamma
        self._setup_logging()
    # [Previous methods remain the same until train]

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _get_loss_function(self, class_weights=None):
        """Return either Focal Loss or weighted CE based on class weights"""
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            return nn.CrossEntropyLoss(weight=class_weights)
        else:
            return FocalLoss(gamma=self.focal_loss_gamma)

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int,
            learning_rate: float = 0.001,
            class_weights: np.ndarray = None,
            early_stopping_patience: int = 5,
            save_path: str = None
    ):
        criterion = self._get_loss_function(class_weights)

        # Two-phase training setup
        optimizers = {
            'frozen_encoder': optim.Adam(self.model.classifier.parameters(), lr=learning_rate),
            'fine_tuning': optim.Adam([
                {'params': self.model.vae_model.parameters(), 'lr': learning_rate * 0.1},
                {'params': self.model.classifier.parameters(), 'lr': learning_rate}
            ])
        }

        schedulers = {
            'frozen_encoder': optim.lr_scheduler.ReduceLROnPlateau(
                optimizers['frozen_encoder'], mode='min', factor=0.5, patience=2
            ),
            'fine_tuning': optim.lr_scheduler.ReduceLROnPlateau(
                optimizers['fine_tuning'], mode='min', factor=0.5, patience=2
            )
        }
        best_val_f1 = 0  # Changed to track F1 instead of loss
        patience_counter = 0
        best_model_state = None
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []  # Added F1 tracking
        }

        self.logger.info("Starting training...")
        for epoch in range(epochs):
            # Training phase
            train_metrics = self._train_epoch(
                train_loader,
                criterion,
                optimizers[self.model.training_phase]
            )

            # Validation phase
            val_metrics = self._validate(val_loader, criterion)

            # Update learning rate
            schedulers[self.model.training_phase].step(val_metrics['loss'])

            # Log progress
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs}\n"
                f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, "
                f"F1: {train_metrics['minority_f1']:.4f}\n"
                f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, "
                f"F1: {val_metrics['minority_f1']:.4f}"
            )

            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['val_loss'].append(val_metrics['loss'])
            history['train_acc'].append(train_metrics['acc'])
            history['val_acc'].append(val_metrics['acc'])
            history['train_f1'].append(train_metrics['minority_f1'])
            history['val_f1'].append(val_metrics['minority_f1'])

            # Early stopping check - now based on F1 score
            if val_metrics['minority_f1'] > best_val_f1:
                best_val_f1 = val_metrics['minority_f1']
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1

            # Phase transition check
            if (patience_counter == early_stopping_patience // 2 and
                    self.model.training_phase == 'frozen_encoder'):
                self.logger.info("Unfreezing encoder for fine-tuning...")
                self.model.unfreeze_encoder()
                patience_counter = 0

            # Early stopping
            elif patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered! Best minority F1: {best_val_f1:.4f}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if save_path:
                torch.save(best_model_state, save_path)

        return history

    def _calculate_metrics(self, labels, predictions):
        """Calculate comprehensive metrics including minority class F1"""
        accuracy = accuracy_score(labels, predictions)
        minority_precision = precision_score(labels, predictions, pos_label=1)
        minority_recall = recall_score(labels, predictions, pos_label=1)
        minority_f1 = f1_score(labels, predictions, pos_label=1)

        return {
            'acc': accuracy,
            'minority_precision': minority_precision,
            'minority_recall': minority_recall,
            'minority_f1': minority_f1
        }

    def _train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            optimizer.zero_grad()
            logits, _ = self.model(inputs)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(train_loader)

        return metrics

    def _validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validating'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                logits, _ = self.model(inputs)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                _, predicted = logits.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds)
        metrics['loss'] = total_loss / len(val_loader)

        return metrics

    def plot_training_history(self, history, save_path=None):
        """Plot training curves including F1 score"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()

        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()

        # F1 plot
        ax3.plot(history['train_f1'], label='Train F1')
        ax3.plot(history['val_f1'], label='Val F1')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score (Minority Class)')
        ax3.set_title('Training and Validation F1 Score')
        ax3.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class ImprovedVAEClassifierTrainer(VAEClassifierTrainer):
    def __init__(self, model, device, focal_loss_gamma=2.0, pos_weight=None):
        super().__init__(model, device, focal_loss_gamma)
        self.pos_weight = pos_weight

    def _get_loss_function(self, class_weights=None):
        """Enhanced loss function with additional balancing"""
        if self.pos_weight is not None:
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]).to(self.device))
        elif class_weights is not None:
            return nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights).to(self.device),
                reduction='mean'
            )
        else:
            return FocalLoss(gamma=self.focal_loss_gamma)

    def train(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int,
            learning_rate: float = 0.001,
            class_weights: np.ndarray = None,
            early_stopping_patience: int = 5,
            save_path: str = None,
            min_recall_threshold: float = 0.3  # Minimum recall we want to achieve
    ):
        criterion = self._get_loss_function(class_weights)

        # Modified optimizer setup with different learning rates
        optimizers = {
            'frozen_encoder': optim.AdamW([
                {'params': self.model.classifier.parameters(), 'lr': learning_rate, 'weight_decay': 0.01}
            ]),
            'fine_tuning': optim.AdamW([
                {'params': self.model.vae_model.parameters(), 'lr': learning_rate * 0.1},
                {'params': self.model.classifier.parameters(), 'lr': learning_rate}
            ], weight_decay=0.01)
        }

        schedulers = {
            'frozen_encoder': optim.lr_scheduler.OneCycleLR(
                optimizers['frozen_encoder'],
                max_lr=learning_rate,
                epochs=epochs,
                steps_per_epoch=len(train_loader)
            ),
            'fine_tuning': optim.lr_scheduler.OneCycleLR(
                optimizers['fine_tuning'],
                max_lr=[learning_rate * 0.1, learning_rate],
                epochs=epochs,
                steps_per_epoch=len(train_loader)
            )
        }

        best_f1_minority = 0
        patience_counter = 0
        best_model_state = None
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'minority_recall': [], 'minority_precision': [],
            'minority_f1': []
        }

        self.logger.info("Starting training with improved minority class handling...")

        for epoch in range(epochs):
            # Training phase with gradient accumulation
            train_loss, train_acc, train_minority_metrics = self._train_epoch(
                train_loader,
                criterion,
                optimizers[self.model.training_phase],
                schedulers[self.model.training_phase],
                grad_accumulation_steps=4  # Effective batch size *= 4
            )

            # Validation phase
            val_loss, val_acc, val_minority_metrics = self._validate(val_loader, criterion)

            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['minority_recall'].append(val_minority_metrics['recall'])
            history['minority_precision'].append(val_minority_metrics['precision'])
            history['minority_f1'].append(val_minority_metrics['f1'])

            # Log detailed metrics
            self.logger.info(
                f"\nEpoch {epoch + 1}/{epochs}:\n"
                f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}\n"
                f"Train Minority - F1: {train_minority_metrics['f1']:.4f}, "
                f"Recall: {train_minority_metrics['recall']:.4f}, "
                f"Precision: {train_minority_metrics['precision']:.4f}\n"
                f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}\n"
                f"Val Minority - F1: {val_minority_metrics['f1']:.4f}, "
                f"Recall: {val_minority_metrics['recall']:.4f}, "
                f"Precision: {val_minority_metrics['precision']:.4f}"
            )

            # Early stopping based on minority class F1
            if val_minority_metrics['f1'] > best_f1_minority:
                best_f1_minority = val_minority_metrics['f1']
                patience_counter = 0
                best_model_state = self.model.state_dict()

                if save_path:
                    torch.save(best_model_state, save_path)
            else:
                patience_counter += 1

            # Phase transition check with minority class performance
            if (patience_counter == early_stopping_patience // 2 and
                    self.model.training_phase == 'frozen_encoder' and
                    val_minority_metrics['recall'] < min_recall_threshold):
                self.logger.info("Unfreezing encoder for fine-tuning...")
                self.model.unfreeze_encoder()
                patience_counter = 0

            # Early stopping
            elif patience_counter >= early_stopping_patience:
                if val_minority_metrics['recall'] >= min_recall_threshold:
                    self.logger.info("Early stopping with satisfactory minority class recall!")
                    break
                elif self.model.training_phase == 'frozen_encoder':
                    self.logger.info("Transitioning to fine-tuning phase...")
                    self.model.unfreeze_encoder()
                    patience_counter = 0
                else:
                    self.logger.info("Early stopping!")
                    break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return history

    def _train_epoch(self, train_loader, criterion, optimizer, scheduler, grad_accumulation_steps=1):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        running_loss = 0
        optimizer.zero_grad()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Forward pass
            logits, _ = self.model(inputs)
            loss = criterion(logits, labels) / grad_accumulation_steps
            loss.backward()

            # Gradient accumulation
            if (i + 1) % grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item() * grad_accumulation_steps
            _, predicted = logits.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        minority_metrics = self._calculate_minority_metrics(all_labels, all_preds)
        accuracy = (all_preds == all_labels).mean()

        return running_loss / len(train_loader), accuracy, minority_metrics

    def _validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                logits, _ = self.model(inputs)
                loss = criterion(logits, labels)
                total_loss += loss.item()
                _, predicted = logits.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        minority_metrics = self._calculate_minority_metrics(all_labels, all_preds)
        accuracy = (all_preds == all_labels).mean()

        return total_loss / len(val_loader), accuracy, minority_metrics

    def _calculate_minority_metrics(self, true_labels, predictions):
        """Calculate metrics focusing on minority class"""
        minority_mask = true_labels == 1
        if not any(minority_mask):
            return {'precision': 0, 'recall': 0, 'f1': 0}

        minority_true = true_labels[minority_mask]
        minority_pred = predictions[minority_mask]

        tp = np.sum((predictions == 1) & (true_labels == 1))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {'precision': precision, 'recall': recall, 'f1': f1}


class TripletAutoencoderTrainer(BaseTrainer):
    def __init__(self, model, criterion, optimizer, scheduler, device,
                 mask_probability=0.1, margin=1.0, triplet_weight=1.0,
                 distance_metric='L2', save_path=None, early_stopping_patience=5):
        super().__init__(model, criterion, optimizer, scheduler, device, save_path, early_stopping_patience)
        self.mask_probability = mask_probability
        self.margin = margin
        self.triplet_weight = triplet_weight
        self.distance_metric = distance_metric
        self.best_val_loss = float('inf')

    def apply_mask(self, inputs):
        """Apply random masking to inputs"""
        if self.mask_probability > 0:
            mask = torch.bernoulli(torch.full(inputs.shape, 1 - self.mask_probability)).to(self.device)
            return inputs * mask
        return inputs

    def compute_triplet_loss(self, embeddings, labels):
        """
        Compute triplet loss with emphasis on minority class
        Args:
            embeddings: Tensor of shape [batch_size, embedding_dim]
            labels: Tensor of shape [batch_size]
        """
        # Create minority mask
        minority_mask = (labels == 1)
        triplet_loss = torch.tensor(0.0, device=embeddings.device)

        if torch.any(minority_mask):
            # Compute pairwise distances
            if self.distance_metric == "L2":
                dist_matrix = torch.cdist(embeddings, embeddings)
            else:  # cosine
                cos_sim_matrix = torch.mm(embeddings, embeddings.t())
                dist_matrix = 1 - cos_sim_matrix

            # Get hardest positive and negative for minority class
            minority_indices = torch.where(minority_mask)[0]
            for anchor_idx in minority_indices:
                pos_mask = (labels == labels[anchor_idx]) & (
                            torch.arange(len(labels), device=labels.device) != anchor_idx)
                neg_mask = labels != labels[anchor_idx]

                if torch.any(pos_mask) and torch.any(neg_mask):
                    # Weight minority class samples more heavily
                    pos_dist = dist_matrix[anchor_idx][pos_mask].mean()
                    neg_dist = dist_matrix[anchor_idx][neg_mask].min()
                    triplet_loss += torch.relu(pos_dist - neg_dist + self.margin) * 2.0  # Higher weight for minority

            # Add some triplets from majority class with lower weight
            majority_indices = torch.where(~minority_mask)[0]
            if len(majority_indices) > 0:
                sampled_majority = majority_indices[torch.randperm(len(majority_indices))[:len(minority_indices)]]
                for anchor_idx in sampled_majority:
                    pos_mask = (labels == labels[anchor_idx]) & (
                                torch.arange(len(labels), device=labels.device) != anchor_idx)
                    neg_mask = labels != labels[anchor_idx]

                    if torch.any(pos_mask) and torch.any(neg_mask):
                        pos_dist = dist_matrix[anchor_idx][pos_mask].mean()
                        neg_dist = dist_matrix[anchor_idx][neg_mask].min()
                        triplet_loss += torch.relu(pos_dist - neg_dist + self.margin)

        return triplet_loss

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_triplet_loss = 0
        num_batches = len(train_loader)

        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            inputs = self.apply_mask(inputs)

            self.optimizer.zero_grad()

            reconstructed, embeddings = self.model(inputs)

            # Compute reconstruction loss
            recon_loss = self.criterion(reconstructed, inputs)

            # Compute triplet loss
            triplet_loss = self.compute_triplet_loss(embeddings, labels)

            # Combine losses
            loss = recon_loss + self.triplet_weight * triplet_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_triplet_loss += triplet_loss.item()

        return {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'triplet_loss': total_triplet_loss / num_batches
        }

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_triplet_loss = 0
        num_batches = len(val_loader)

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs = self.apply_mask(inputs)

                # Forward pass with embeddings
                reconstructed, embeddings = self.model(inputs, return_embeddings=True)

                # Compute losses
                recon_loss = self.criterion(reconstructed, inputs)
                triplet_loss = self.compute_triplet_loss(embeddings, labels)
                loss = recon_loss + self.triplet_weight * triplet_loss

                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_triplet_loss += triplet_loss.item()

        return {
            'total_loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'triplet_loss': total_triplet_loss / num_batches
        }

    def plot_embedding_space(self, loader, epoch=None):
        """Plot the embedding space using t-SNE"""
        self.model.eval()
        embeddings_list = []
        labels_list = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                embeddings = self.model.get_normalized_embeddings(inputs)
                embeddings_list.append(embeddings.cpu().numpy())
                labels_list.append(labels.numpy())

        embeddings = np.concatenate(embeddings_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Plot
        plt.figure(figsize=(10, 8))
        for label in np.unique(labels):
            mask = labels == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                        label=f'Class {label}', alpha=0.6)
        plt.legend()
        plt.title('Embedding Space Visualization (t-SNE)')
        if epoch is not None:
            plt.savefig(os.path.join(self.save_path, f'embeddings_epoch_{epoch}.png'))
        else:
            plt.savefig(os.path.join(self.save_path, 'embeddings.png'))
        plt.close()

    def train(self, train_loader, val_loader, epochs):
        """Main training loop with visualization"""
        train_losses = {'total_loss': [], 'recon_loss': [], 'triplet_loss': []}
        val_losses = {'total_loss': [], 'recon_loss': [], 'triplet_loss': []}
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            for k, v in train_metrics.items():
                train_losses[k].append(v)

            # Validation
            val_metrics = self.validate(val_loader)
            for k, v in val_metrics.items():
                val_losses[k].append(v)

            # Plot embeddings periodically
            if epoch % 10 == 0:
                self.plot_embedding_space(val_loader, epoch)

            # Early stopping check
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                patience_counter = 0
                if self.save_path:
                    self.save_checkpoint(self.save_path, epoch, best=True)
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            # Print progress
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Train - Total: {train_metrics['total_loss']:.4f}, "
                  f"Recon: {train_metrics['recon_loss']:.4f}, "
                  f"Triplet: {train_metrics['triplet_loss']:.4f}")
            print(f"Val - Total: {val_metrics['total_loss']:.4f}, "
                  f"Recon: {val_metrics['recon_loss']:.4f}, "
                  f"Triplet: {val_metrics['triplet_loss']:.4f}")

        # Final embedding visualization
        self.plot_embedding_space(val_loader)
        return train_losses, val_losses


class EnsembleTrainer:
    def __init__(self,
                 base_model_class,
                 model_params: dict,
                 n_models: int = 5,
                 device: str = 'cuda',
                 save_path: str = None):

        # Create save directory if it doesn't exist
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            print(f"Save directory created/verified at: {save_path}")
        self.base_model_class = base_model_class
        self.model_params = model_params
        self.n_models = n_models
        self.device = device
        self.save_path = save_path
        self.models = []
        self.best_val_losses = []

    def create_weighted_loader(self, dataset, batch_size, majority_weight=0.1):
        """Create a DataLoader with weighted sampling (lower weight for majority class)"""
        # Extract labels from TensorDataset
        labels = dataset.tensors[1].cpu().numpy()

        # Calculate class weights
        unique_labels, counts = np.unique(labels, return_counts=True)
        minority_idx = counts.argmin()

        # Assign weights
        weights = np.ones(len(labels))
        majority_mask = labels != unique_labels[minority_idx]
        weights[majority_mask] = majority_weight  # Lower weight for majority class

        # Normalize weights
        weights = weights / weights.sum()

        # Create sampler
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )

        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    def create_undersampled_loader(self, dataset, batch_size):
        """Create a DataLoader with undersampling of majority class"""
        # Extract labels from TensorDataset
        labels = dataset.tensors[1].cpu().numpy()
        data = dataset.tensors[0].cpu().numpy()

        # Find minority and majority indices
        unique_labels, counts = np.unique(labels, return_counts=True)
        minority_class = unique_labels[counts.argmin()]
        minority_count = counts.min()

        # Get indices for each class
        minority_indices = np.where(labels == minority_class)[0]
        majority_indices = np.where(labels != minority_class)[0]

        # Randomly sample from majority class to match minority class size
        sampled_majority_indices = np.random.choice(
            majority_indices,
            size=minority_count,
            replace=False
        )

        # Combine indices and shuffle
        selected_indices = np.concatenate([minority_indices, sampled_majority_indices])
        np.random.shuffle(selected_indices)

        # Create balanced dataset
        balanced_data = data[selected_indices]
        balanced_labels = labels[selected_indices]

        # Convert back to tensors
        balanced_dataset = TensorDataset(
            torch.FloatTensor(balanced_data),
            torch.LongTensor(balanced_labels)
        )

        return DataLoader(balanced_dataset, batch_size=batch_size, shuffle=True)
    def train_single_model(self, model, train_loader, val_loader, epochs, criterion, optimizer):
        """Train a single model"""
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_minority_f1 = 0
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    _, preds = torch.max(output, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(target.cpu().numpy())

            val_loss /= len(val_loader)
            minority_f1 = f1_score(val_targets, val_preds, pos_label=1, average='binary')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_minority_f1 = minority_f1
                patience_counter = 0
                if self.save_path:
                    torch.save(model.state_dict(),
                               f"{self.save_path}/model_{len(self.models)}.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return model, best_val_loss,best_minority_f1

    def train_ensemble(self, train_dataset, val_loader, batch_size, epochs, criterion,
                       optimizer_class, optimizer_params, majority_weight=0.01):
        """Train the ensemble of models using weighted sampling"""
        self.models = []
        self.best_val_losses = []

        for i in tqdm(range(self.n_models), desc="Training ensemble"):
            # Create new model instance
            model = self.base_model_class(**self.model_params).to(self.device)
            optimizer = optimizer_class(model.parameters(), **optimizer_params)

            # Create weighted loader for this model
            # train_loader = self.create_weighted_loader(
            #     train_dataset, batch_size, majority_weight)
            train_loader = self.create_undersampled_loader(
                train_dataset, batch_size)

            # Train model
            trained_model, best_val_loss,minority_f1 = self.train_single_model(
                model, train_loader, val_loader, epochs, criterion, optimizer)

            self.models.append(trained_model)
            self.best_val_losses.append(best_val_loss)

            print(f"Model {i+1}/{self.n_models} trained. Best val loss: {best_val_loss:.4f}, "
                  f"Minority class F1: {minority_f1:.4f}")

    def predict(self, data_loader, minority_weight=1.5):
        """
        Make predictions using weighted majority voting to favor minority class.

        Args:
            data_loader: DataLoader with test data
            minority_weight: Weight to apply to minority class predictions (>1 favors minority class)
        """
        all_predictions = []
        all_probabilities = []  # Store probabilities for each model

        # Get predictions and probabilities from each model
        for model in self.models:
            model.eval()
            predictions = []
            probabilities = []

            with torch.no_grad():
                for data, _ in data_loader:
                    data = data.to(self.device)
                    outputs = model(data)
                    probs = F.softmax(outputs, dim=1)

                    # Store both predictions and probabilities
                    _, preds = torch.max(outputs, 1)
                    predictions.extend(preds.cpu().numpy())
                    probabilities.extend(probs.cpu().numpy())

            all_predictions.append(predictions)
            all_probabilities.append(probabilities)

        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)  # shape: [n_models, n_samples]
        all_probabilities = np.array(all_probabilities)  # shape: [n_models, n_samples, n_classes]

        # Apply weighted voting
        final_predictions = []
        for i in range(all_predictions.shape[1]):  # For each sample
            # Get predictions for this sample from all models
            sample_preds = all_predictions[:, i]

            # Weight minority class predictions
            minority_votes = np.sum(sample_preds == 1)
            majority_votes = np.sum(sample_preds == 0)

            weighted_minority = minority_votes * minority_weight
            weighted_majority = majority_votes

            final_pred = 1 if weighted_minority > weighted_majority else 0
            final_predictions.append(final_pred)

        return np.array(final_predictions)

    def predict_proba(self, data_loader):
        """
        Get probability predictions from ensemble.
        Returns average probability across all models.
        """
        all_probabilities = []

        for model in self.models:
            model.eval()
            model_probs = []

            with torch.no_grad():
                for data, _ in data_loader:
                    data = data.to(self.device)
                    outputs = model(data)
                    probs = F.softmax(outputs, dim=1)
                    model_probs.extend(probs.cpu().numpy())

            all_probabilities.append(model_probs)

        # Average probabilities across models
        return np.mean(all_probabilities, axis=0)

    def evaluate(self, test_loader):
        """Evaluate the ensemble"""
        predictions = self.predict(test_loader)

        true_labels = []
        for _, labels in test_loader:
            true_labels.extend(labels.cpu().numpy())

        for weight in [1.2, 1.5, 2.0, 2.5,5]:
            predictions = self.predict(test_loader, minority_weight=weight)
            f1 = f1_score(true_labels, predictions, average='binary')
            print(f"\nMinority Weight: {weight}")
            print(f"F1 Score: {f1:.4f}")
            print(confusion_matrix(true_labels, predictions))

        accuracy = accuracy_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions, average='weighted')

        print(f"\nEnsemble Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        cm= confusion_matrix(true_labels,predictions)
        print(cm)
        report = classification_report(true_labels, predictions)

        if self.save_path:
            with open(os.path.join(self.save_path, 'evaluation_results.txt'), 'w') as f:
                f.write("Classification Report:\n")
                f.write(report)
                f.write("\nConfusion Matrix:\n")
                f.write("   Predicted 0  Predicted 1\n")
                f.write(f"Actual 0   {cm[0, 0]:<10} {cm[0, 1]:<10}\n")
                f.write(f"Actual 1   {cm[1, 0]:<10} {cm[1, 1]:<10}\n")
        return accuracy, f1