import itertools
import json
import math

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
from datetime import datetime

from scan_classifer.main_scan import prepare_data, EyeTrackingTokenizer
import os

import pandas as pd
from sklearn.metrics import f1_score, recall_score, confusion_matrix, classification_report

from representation_method.utils.data_loader import load_eye_tracking_data, DataConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from sklearn.preprocessing import KBinsDiscretizer

from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification

from representation_method.utils.general_utils import seed_everything


class ImprovedEyeTrackingTransformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            n_features: int,
            d_model: int = 256,
            n_heads: int = 8,
            n_layers: int = 6,
            dropout: float = 0.1,
            max_len: int = 1000,
            pretrained_model: str = "xlnet-base-cased",
            gradient_checkpointing: bool = True
    ):
        super().__init__()

        # Ensure d_model is divisible by both n_heads and n_features
        d_model = (d_model // n_features // n_heads) * n_features * n_heads
        self.d_model = d_model
        self.feature_dim = d_model // n_features

        # Enhanced token embeddings with layer normalization
        self.token_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Embedding(vocab_size, self.feature_dim),
                nn.LayerNorm(self.feature_dim)
            )
            for _ in range(n_features)
        ])

        # Improved position encoding with learnable weights
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
        self.pos_dropout = nn.Dropout(dropout)

        # Load and configure XLNet
        config = AutoConfig.from_pretrained(pretrained_model)
        config.num_attention_heads = n_heads
        config.hidden_size = d_model
        config.num_hidden_layers = n_layers
        config.dropout = dropout
        config.attention_dropout = dropout
        config.gradient_checkpointing = gradient_checkpointing

        # Initialize transformer with custom config
        self.transformer = AutoModelForSequenceClassification.from_config(config)

        # Add auxiliary heads for multitask learning
        self.sequence_length_predictor = nn.Linear(d_model, 1)
        self.feature_reconstructor = nn.Linear(d_model, n_features)

        # Attention pooling layer
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )

        # Final classification layer with residual connection
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Doubled input size for concatenated features
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)  # Binary classification
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        batch_size, seq_len, n_features = x.shape

        # Enhanced feature embedding
        embeddings = []
        for i, embedding_layer in enumerate(self.token_embeddings):
            feature_tokens = x[:, :, i]
            feature_embedding = embedding_layer(feature_tokens)
            embeddings.append(feature_embedding)

        # Concatenate and apply position encoding
        x = torch.cat(embeddings, dim=-1)
        pos_embedding = self.pos_embedding[:, :seq_len, :]
        x = x + pos_embedding
        x = self.pos_dropout(x)

        # Create attention mask for valid tokens
        attention_mask = torch.ones((batch_size, seq_len), device=x.device)

        # Forward through transformer
        transformer_outputs = self.transformer(
            inputs_embeds=x,
            attention_mask=attention_mask,
            output_attentions=return_attention,
            return_dict=True
        )

        # Get sequence representation using attention pooling
        attention_weights = self.attention_pool(transformer_outputs.last_hidden_state)
        weighted_sum = torch.sum(attention_weights * transformer_outputs.last_hidden_state, dim=1)

        # Get max pooled features
        max_pooled = torch.max(transformer_outputs.last_hidden_state, dim=1)[0]

        # Concatenate different pooling strategies
        combined_features = torch.cat([weighted_sum, max_pooled], dim=-1)

        # Final classification
        logits = self.classifier(combined_features)

        if return_attention:
            return logits, attention_weights
        return logits

    def compute_auxiliary_losses(self, x: torch.Tensor, transformer_outputs) -> Dict[str, torch.Tensor]:
        # Sequence length prediction
        pred_length = self.sequence_length_predictor(transformer_outputs.last_hidden_state.mean(dim=1))
        true_length = (x != 0).sum(dim=1).float().unsqueeze(-1)
        length_loss = F.mse_loss(pred_length, true_length)

        # Feature reconstruction
        reconstructed_features = self.feature_reconstructor(transformer_outputs.last_hidden_state)
        reconstruction_loss = F.mse_loss(reconstructed_features, x)

        return {
            'length_loss': length_loss,
            'reconstruction_loss': reconstruction_loss
        }


def train_improved_model(
        model: ImprovedEyeTrackingTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        learning_rate: float = 1e-4,
        warmup_steps: int = 1000,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        aux_loss_weight: float = 0.1
):
    # Initialize AdamW with weight decay
    device = 'cuda'
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    # Main training loop
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for step, (tokens, labels) in enumerate(train_loader):
            tokens, labels = tokens.to(device), labels.to(device)

            # Forward pass with auxiliary tasks
            outputs = model(tokens)
            aux_losses = model.compute_auxiliary_losses(tokens, outputs)

            # Compute main classification loss
            main_loss = F.cross_entropy(outputs.logits, labels)

            # Combine losses
            total_loss = main_loss + aux_loss_weight * (
                    aux_losses['length_loss'] + aux_losses['reconstruction_loss']
            )

            # Gradient accumulation
            total_loss = total_loss / gradient_accumulation_steps
            total_loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += total_loss.item() * gradient_accumulation_steps

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for tokens, labels in val_loader:
                tokens, labels = tokens.to(device), labels.to(device)
                outputs = model(tokens)
                loss = F.cross_entropy(outputs.logits, labels)
                val_loss += loss.item()

                val_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

        # Print metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print(f'Val F1: {val_f1:.4f}')
        print(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')


def get_cosine_schedule_with_warmup(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
class GridSearchOptimizer:
    def __init__(
            self,
            base_save_dir: str = "grid_search_results",
            seeds: List[int] = [42, 66, 93, 123, 456],
            device: str = None
    ):
        self.base_save_dir = Path(base_save_dir)
        self.seeds = seeds
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._setup_logging()

        # Define parameter grid
        self.param_grid = {
            'model_params': {
                'd_model': [128, 256],
                'n_heads': [8, 16],
                'n_layers': [4, 6],
                'dropout': [0.1, 0.2, 0.3],
            },
            'training_params': {
                'learning_rate': [1e-4, 3e-4, 5e-4],
                'warmup_steps': [500, 1000],
                'gradient_accumulation_steps': [4, 8],
                'aux_loss_weight': [0.05, 0.1, 0.15]
            }
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('GridSearch')
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

            # Console handler
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)

            # File handler
            self.base_save_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(
                self.base_save_dir / f'grid_search_{datetime.now():%Y%m%d_%H%M%S}.log'
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger

    def generate_param_combinations(self) -> List[Dict]:
        """Generate all possible parameter combinations"""
        model_keys, model_values = zip(*self.param_grid['model_params'].items())
        training_keys, training_values = zip(*self.param_grid['training_params'].items())

        combinations = []
        for model_combo in itertools.product(*model_values):
            for training_combo in itertools.product(*training_values):
                param_set = {
                    'model_params': dict(zip(model_keys, model_combo)),
                    'training_params': dict(zip(training_keys, training_combo))
                }
                combinations.append(param_set)

        return combinations

    def prepare_data_with_seed(self, df: pd.DataFrame, feature_columns: List[str],
                               tokenizer_class, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader, Any]:
        """Prepare data splits using the specified seed"""
        # Set seed for reproducibility of data splitting
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Split data
        train_loader, val_loader, test_loader, tokenizer = prepare_data(
            df=df,
            feature_columns=feature_columns,
            test_size=0.15,
            val_size=0.15,
            seed=seed
        )

        return train_loader, val_loader, test_loader, tokenizer

    def train_and_evaluate(
            self,
            param_set: Dict,
            train_loader,
            val_loader,
            test_loader,
            tokenizer,
            feature_columns: List[str],
            tokenizer_class,
            n_features: int,
            seed: int
    ) -> Dict:
        """Train and evaluate a single parameter set with given seed"""
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Initialize model with current parameters
        model = ImprovedEyeTrackingTransformer(
            vocab_size=tokenizer.vocab_size,
            n_features=n_features,
            **param_set['model_params'],
            pretrained_model='xlnet-base-cased',
            gradient_checkpointing=True
        ).to(self.device)

        # Prepare data with the current seed


        # Train model
        model_save_dir = self.base_save_dir / f"model_seed_{seed}"
        model_save_dir.mkdir(exist_ok=True)

        train_improved_model(
            model=model,train_loader = train_loader,val_loader= val_loader,
            **param_set['training_params']
        )

        # Evaluate on test set
        test_metrics = self._evaluate_model(model, test_loader)

        return {
            'test_metrics': test_metrics,
            'seed': seed
        }

    def _evaluate_model(self, model: torch.nn.Module, test_loader: DataLoader) -> Dict:
        """Evaluate model on test set"""
        model.eval()
        all_preds = []
        all_labels = []
        test_loss = 0

        with torch.no_grad():
            for tokens, labels in test_loader:
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)

                outputs = model(tokens)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                test_loss += loss.item()

                preds = outputs.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = {
            'test_loss': test_loss / len(test_loader),
            'accuracy': (np.array(all_preds) == np.array(all_labels)).mean(),
            'f1': f1_score(all_labels, all_preds, average='weighted'),
            'precision': precision_score(all_labels, all_preds, average='weighted'),
            'recall': recall_score(all_labels, all_preds, average='weighted')
        }

        return metrics

    def run_grid_search(
            self,
            df: pd.DataFrame,
            feature_columns: List[str],
            tokenizer_class,
            n_features: int
    ) -> pd.DataFrame:
        """Run complete grid search"""
        param_combinations = self.generate_param_combinations()
        self.logger.info(f"Starting grid search with {len(param_combinations)} parameter combinations")

        all_results = []

        for param_idx, param_set in enumerate(param_combinations):
            self.logger.info(f"\nParameter combination {param_idx + 1}/{len(param_combinations)}")
            self.logger.info(f"Parameters: {param_set}")

            # Run for each seed
            seed_results = []
            for seed in self.seeds:
                self.logger.info(f"Training with seed {seed}")
                try:
                    # Prepare data with current seed
                    train_loader, val_loader, test_loader, tokenizer = self.prepare_data_with_seed(
                        df, feature_columns, tokenizer_class, seed
                    )

                    result = self.train_and_evaluate(
                    param_set,
                    train_loader,
                    val_loader,
                    test_loader,
                    tokenizer,
                    feature_columns,
                    tokenizer_class,
                    n_features,
                    seed)
                    seed_results.append(result)
                except Exception as e:
                    self.logger.error(f"Error with seed {seed}: {str(e)}")
                    continue

            # Calculate statistics across seeds
            metrics = self._calculate_seed_statistics(seed_results)

            # Store results
            result_entry = {
                'param_set': param_set,
                'mean_f1': metrics['f1_mean'],
                'f1_std': metrics['f1_std'],
                'mean_accuracy': metrics['accuracy_mean'],
                'accuracy_std': metrics['accuracy_std'],
                'stability_score': metrics['stability_score'],
                'seed_results': seed_results
            }
            all_results.append(result_entry)

            # Save intermediate results
            self._save_results(all_results)

        # Convert to DataFrame and sort by stability score
        results_df = self._create_results_dataframe(all_results)
        return results_df

    def _calculate_seed_statistics(self, seed_results: List[Dict]) -> Dict:
        """Calculate statistics across seeds"""
        f1_scores = [r['test_metrics']['f1'] for r in seed_results]
        accuracies = [r['test_metrics']['accuracy'] for r in seed_results]

        # Calculate means and stds
        f1_mean = np.mean(f1_scores)
        f1_std = np.std(f1_scores)
        accuracy_mean = np.mean(accuracies)
        accuracy_std = np.std(accuracies)

        # Calculate stability score (higher is better)
        # Combines high mean performance with low variance
        stability_score = (f1_mean * accuracy_mean) / (1 + f1_std + accuracy_std)

        return {
            'f1_mean': f1_mean,
            'f1_std': f1_std,
            'accuracy_mean': accuracy_mean,
            'accuracy_std': accuracy_std,
            'stability_score': stability_score
        }

    def _save_results(self, results: List[Dict]):
        """Save results to file"""
        save_path = self.base_save_dir / 'grid_search_results.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

    def _create_results_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Create DataFrame from results"""
        df_data = []
        for result in results:
            entry = {
                'stability_score': result['stability_score'],
                'mean_f1': result['mean_f1'],
                'f1_std': result['f1_std'],
                'mean_accuracy': result['mean_accuracy'],
                'accuracy_std': result['accuracy_std']
            }
            # Add model parameters
            entry.update({f"model_{k}": v for k, v in result['param_set']['model_params'].items()})
            # Add training parameters
            entry.update({f"train_{k}": v for k, v in result['param_set']['training_params'].items()})
            df_data.append(entry)

        df = pd.DataFrame(df_data)
        return df.sort_values('stability_score', ascending=False)


def run_grid_search_optimization(
        df: pd.DataFrame,
        feature_columns: List[str],
        tokenizer_class,
        n_features: int,
        base_save_dir: str = "grid_search_results"
):
    """Main function to run grid search optimization"""
    grid_search = GridSearchOptimizer(base_save_dir=base_save_dir)

    # Run grid search
    results_df = grid_search.run_grid_search(
        df=df,
        feature_columns=feature_columns,
        tokenizer_class=tokenizer_class,
        n_features=n_features
    )

    # Save final results
    results_df.to_csv(Path(base_save_dir) / 'final_results.csv', index=False)

    # Print top 5 configurations
    print("\nTop 5 Most Stable Configurations:")
    print(results_df.head())

    return results_df

if __name__ == '__main__':


    config = DataConfig(
        data_path='data/Categorized_Fixation_Data_1_18.csv',
        approach_num=6,
        normalize=True,
        per_slice_target=True,
        participant_id=None
    )

    df = load_eye_tracking_data(
        data_path=config.data_path,
        approach_num=config.approach_num,
        participant_id=config.participant_id,
        data_format="legacy"
    )


    # Define feature columns
    feature_columns = [
        'Pupil_Size',
        'CURRENT_FIX_DURATION',
        'CURRENT_FIX_INDEX',
        'CURRENT_FIX_COMPONENT_COUNT',
    ]

    results_df = run_grid_search_optimization(
        df=df,
        feature_columns=feature_columns,
        tokenizer_class=EyeTrackingTokenizer,  # Your tokenizer class
        n_features=len(feature_columns) + 2,
        base_save_dir="grid_search_results"
    )