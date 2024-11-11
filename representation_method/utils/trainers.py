import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
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

    def save_model(self, path, epoch=None, best=False):
        """Save model checkpoint"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'epoch': epoch
        }
        filename = 'best_model.pth' if best else f'checkpoint_epoch_{epoch}.pth'
        torch.save(save_dict, os.path.join(path, filename))

    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint.get('epoch', 0)


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
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs):
        """Main training loop"""
        train_losses = []
        val_losses = []
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if self.scheduler:
                self.scheduler.step(val_loss)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                if self.save_path:
                    self.save_model(self.save_path, epoch, best=True)
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        # Plot training curves
        if self.save_path:
            self._plot_training_curves(train_losses, val_losses)

    def _plot_training_curves(self, train_losses, val_losses):
        """Plot and save training curves"""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'training_curves.png'))
        plt.close()


class CombinedModelTrainer(BaseTrainer):
    def train_epoch(self, train_loader, window_weights=None):
        """Train for one epoch with progress bar"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc='Training', leave=False)
        for i, (inputs, labels) in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            weights = window_weights[i].to(self.device) if window_weights is not None else None

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

            # Update progress bar
            pbar.set_postfix({
                'batch_loss': f'{loss.item():.4f}',
                'batch_acc': f'{100. * batch_correct / batch_total:.2f}%'
            })

        return total_loss / len(train_loader), correct / total


    def validate(self, val_loader, window_weights=None):
        """Validate with progress bar"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(enumerate(val_loader), total=len(val_loader),
                    desc='Validating', leave=False)
        with torch.no_grad():
            for i, (inputs, labels) in pbar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                weights = window_weights[i].to(self.device) if window_weights is not None else None

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

                # Update progress bar
                pbar.set_postfix({
                    'batch_loss': f'{loss.item():.4f}',
                    'batch_acc': f'{100. * batch_correct / batch_total:.2f}%'
                })

        return total_loss / len(val_loader), correct / total

    def evaluate(self, test_loader, threshold=0.5):
        """Evaluate with progress bar"""
        self.model.eval()
        all_preds = []
        all_labels = []

        pbar = tqdm(test_loader, desc='Evaluating')
        with torch.no_grad():
            for inputs, labels in pbar:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)[:, 0]
                probabilities = torch.sigmoid(outputs).squeeze()
                preds = (probabilities >= threshold).long()

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Calculate running accuracy
                running_acc = np.mean(np.array(all_preds) == np.array(all_labels))
                pbar.set_postfix({'running_acc': f'{100. * running_acc:.2f}%'})

        report = classification_report(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)

        if self.save_path:
            with open(os.path.join(self.save_path, 'evaluation_results.txt'), 'w') as f:
                f.write("Classification Report:\n")
                f.write(report)
                f.write("\nConfusion Matrix:\n")
                f.write("   Predicted 0  Predicted 1\n")
                f.write(f"Actual 0   {cm[0, 0]:<10} {cm[0, 1]:<10}\n")
                f.write(f"Actual 1   {cm[1, 0]:<10} {cm[1, 1]:<10}\n")

        return report, cm

    def train(self, train_loader, val_loader, epochs, window_weights_train=None, window_weights_val=None):
        """Main training loop with epoch progress bar"""
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        patience_counter = 0

        # Main epoch progress bar
        epoch_pbar = tqdm(range(epochs), desc='Training Progress')
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
                    self.save_model(self.save_path, epoch, best=True)
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break

        if self.save_path:
            self._plot_training_curves(train_losses, val_losses, train_accs, val_accs)


    def _plot_training_curves(self, train_losses, val_losses, train_accs, val_accs):
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
        plt.savefig(os.path.join(self.save_path, 'training_curves.png'))
        plt.close()