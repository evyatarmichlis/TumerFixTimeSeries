import heapq
import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, recall_score, confusion_matrix, classification_report

from representation_method.utils.data_loader import load_eye_tracking_data, DataConfig
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification

from representation_method.utils.general_utils import seed_everything
from scan_classifer.model_interpertaer import TransformerInterpreter

class PreTrainedEyeTrackingTransformer(nn.Module):
    """
    Modified EyeTrackingTransformer using pre-trained transformer backbone
    while maintaining the same interface as original model
    """
    def __init__(
            self,
            vocab_size: int,
            n_features: int,
            d_model: int = 256,
            n_heads: int = 8,
            n_layers: int = 6,
            dropout: float = 0.5,
            max_len: int = 1700,
            pretrained_model: str = "bert-base-uncased"
    ):
        super().__init__()

        # Ensure d_model is divisible by both n_heads and n_features
        d_model = (d_model // n_features // n_heads) * n_features * n_heads
        self.d_model = d_model
        self.feature_dim = d_model // n_features

        # Token embeddings for each feature (maintain original interface)
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, self.feature_dim)
            for _ in range(n_features)
        ])

        # Position encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

        # Load pre-trained transformer config and adjust for our task
        config = AutoConfig.from_pretrained(pretrained_model)
        config.num_attention_heads = n_heads
        config.hidden_size = d_model
        config.num_hidden_layers = n_layers
        config.dropout = dropout
        config.attn_implementation = "eager"
        if pretrained_model != "xlnet-base-cased":
            config.max_position_embeddings = max_len

        # Initialize pre-trained transformer with custom config
        # self.transformer = AutoModel.from_config(config)# baseline
        self.transformer = AutoModelForSequenceClassification.from_config(config)
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, x: torch.Tensor, output_attentions: bool = False) -> tuple[Any, Any] | Any:
        # x shape: [batch_size, seq_len, n_features]
        batch_size, seq_len, n_features = x.shape

        # Embed each feature separately (maintain original interface)
        embeddings = []
        for i, embedding_layer in enumerate(self.token_embeddings):
            feature_tokens = x[:, :, i].long()
            feature_embedding = embedding_layer(feature_tokens)  # [batch_size, seq_len, feature_dim]
            embeddings.append(feature_embedding)

        # Concatenate feature embeddings
        x = torch.cat(embeddings, dim=-1)  # [batch_size, seq_len, d_model]

        # Add positional embeddings
        # x = x + self.pos_embedding[:, :seq_len, :]

        # Create attention mask (all tokens are valid)
        attention_mask = (x != -1).any(dim=-1).float()  # [batch_size, seq_len]
        # Pass through transformer
        outputs = self.transformer(
            inputs_embeds=x,
            attention_mask=attention_mask,
            return_dict=True
        )
        # If output_attentions is True, return both logits and attention weights
        if output_attentions:
            return outputs.logits, outputs.attentions

        else:
            return outputs.logits

        # # Global average pooling on sequence dimension
        # x = transformer_outputs.last_hidden_state.mean(dim=1)
        #
        # # Classification
        # return self.classifier(x)



class EyeTrackingTokenizer:
    def __init__(self, n_bins: int = 20, strategy: str = 'quantile'):
        self.n_bins = n_bins
        self.strategy = strategy
        self.discretizers = {}
        self.vocab_size = None
        self.feature_offsets = {}

    def fit(self, df: pd.DataFrame, feature_columns: List[str]):
        current_offset = 0
        for feature in feature_columns:
            discretizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode='ordinal',
                strategy=self.strategy
            )
            values = df[feature].values.reshape(-1, 1)
            discretizer.fit(values)

            self.discretizers[feature] = discretizer
            self.feature_offsets[feature] = current_offset
            current_offset += self.n_bins

        self.vocab_size = current_offset
        return self

    def tokenize(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        tokens_list = []
        for feature in feature_columns:
            values = df[feature].values.reshape(-1, 1)
            discretizer = self.discretizers[feature]
            offset = self.feature_offsets[feature]
            tokens = discretizer.transform(values) + offset
            tokens_list.append(tokens)
        return np.column_stack(tokens_list)  # Will maintain sequence length


class ScanDataset(Dataset):
    def __init__(self, sequences, labels, sequence_ids, target_positions=None, max_length=1700):
        self.sequences = sequences # Now directly stores the tensor
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sequence_ids = sequence_ids
        self.target_positions = target_positions if target_positions else [[] for _ in range(len(sequences))]

        # Pad sequences if needed
        if self.sequences.size(1) > max_length:
            self.sequences = self.sequences[:, :max_length, :]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.sequence_ids[idx], self.target_positions[idx]

class EyeTrackingDataset(Dataset):
    def __init__(
        self,
        tokens: np.ndarray,
        labels: np.ndarray,
        sequence_ids: List[Tuple[int, int]],
        target_positions: List[List[int]] = None,  # Optional to maintain backward compatibility
        max_length: int = 1700
    ):
        self.tokens = torch.LongTensor(tokens)
        self.labels = torch.LongTensor(labels)
        self.sequence_ids = sequence_ids
        self.target_positions = target_positions if target_positions is not None else [[] for _ in range(len(tokens))]

        if self.tokens.size(1) > max_length:
            self.tokens = self.tokens[:, :max_length]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return (
            self.tokens[idx],
            self.labels[idx],
            self.sequence_ids[idx],
            self.target_positions[idx]
        )


def create_model(vocab_size: int, n_features: int, d_model: int = 256, pretrained_model: str = "roberta-base"):
    """Create model with automatic dimension adjustment"""
    # Ensure d_model is divisible by both n_features and n_heads
    n_heads = 8  # Default number of heads
    d_model = ((d_model // n_features // n_heads) * n_features * n_heads)

    model = PreTrainedEyeTrackingTransformer(
        vocab_size=vocab_size,
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        pretrained_model=pretrained_model
    )


    # Configure model to output attention weights
    model.transformer.config.output_attentions = True
    print(f"\nModel Architecture:")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of features: {n_features}")
    print(f"Model dimension (d_model): {d_model}")
    print(f"Feature embedding dimension: {d_model // n_features}")
    print(f"Number of heads: {n_heads}")
    print(f"Pre-trained model: {pretrained_model}")

    return model


def prepare_data(df: pd.DataFrame, feature_columns: List[str], test_size: float = 0.15, val_size: float = 0.15, seed=0):
    # Group by trial and prepare data
    scans_data = []
    scans_labels = []
    target_positions = []
    scans_ids = []
    sequence_lengths = []
    sequence_ids = []

    # Analysis counters
    total_scans = 0
    abnormal_scans = 0
    scans_with_targets = 0
    abnormal_with_targets = 0
    normal_with_targets = 0
    target_counts = []

    # Group by RECORDING_SESSION_LABEL and TRIAL_INDEX
    for (participant_id, trial_id), group in df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']):
        if len(group) == 1:
            continue

        total_scans += 1
        sequence_ids.append((participant_id, trial_id))
        scans_data.append(group[feature_columns].values)
        sequence_lengths.append(len(group))

        # Store scan type label (normal vs abnormal)
        label = (group['SCAN_TYPE'] != 'NORMAL').any().astype(int)
        # label = (group['target'] == 1).any().astype(int)  # 1 if any target exists in scan, 0 otherwise
        if label == 1:
            abnormal_scans += 1
        scans_labels.append(label)

        # Store and analyze target positions
        has_targets = False
        if 'target' in group.columns:
            target_pos = group.index[group['target'] == 1].tolist()
            seq_start = group.index[0]
            target_pos = [pos - seq_start for pos in target_pos]
            target_positions.append(target_pos)

            if len(target_pos) > 0:
                has_targets = True
                scans_with_targets += 1
                target_counts.append(len(target_pos))
                if label == 1:
                    abnormal_with_targets += 1
                else:
                    normal_with_targets += 1
        else:
            target_positions.append([])

        # Calculate dynamic features
        diff_features = []
        key_features = ['Pupil_Size', 'CURRENT_FIX_DURATION']
        for feature in key_features:
            values = group[feature].values
            diffs = np.diff(values)
            if diffs.size > 0:
                max_diff = np.abs(np.max(np.abs(diffs)))
            else:
                max_diff = 0
            diff_features.append(max_diff)

        diff_features = np.array(diff_features)
        diff_features = np.tile(diff_features, (len(group), 1))
        scans_data[-1] = np.concatenate((scans_data[-1], diff_features), axis=1)
        scans_ids.append(trial_id)

    # Find max sequence length for padding
    max_seq_length = max(sequence_lengths)
    print(f"\nSequence length statistics:")
    print(f"Max length: {max_seq_length}")
    print(f"Min length: {min(sequence_lengths)}")
    print(f"Mean length: {np.mean(sequence_lengths):.2f}")

    # Convert to arrays
    scans_labels = np.array(scans_labels)
    scans_ids = np.array(scans_ids)

    # Initialize and fit tokenizer
    feature_columns = feature_columns + ["diff pupil", "diff fix duration"]
    tokenizer = EyeTrackingTokenizer(n_bins=20, strategy='quantile')
    all_features = pd.DataFrame(np.vstack(scans_data), columns=feature_columns)
    tokenizer.fit(all_features, feature_columns)

    def pad_sequence(sequence: np.ndarray, max_length: int) -> np.ndarray:
        """Pad sequence to max_length"""
        pad_length = max_length - len(sequence)
        if pad_length > 0:
            # Pad with zeros
            padded = np.pad(sequence, ((0, pad_length), (0, 0)), mode='constant', constant_values=-1)
            return padded
        return sequence

    def create_dataset(indices, scans_data, scans_labels, sequence_ids, tokenizer, feature_columns, max_seq_length,target_positions):
        sequences = [scans_data[i] for i in indices]
        sequence_labels = scans_labels[indices]
        sequence_ids = [sequence_ids[i] for i in indices]
        print(f"\nCreating dataset:")
        print(f"Number of sequences: {len(sequences)}")
        print(f"Number of labels: {len(sequence_labels)}")
        print("Trials in this dataset:")
        trails =  set(sorted([trial_id for participant_id, trial_id in sequence_ids]))
        print(trails)
        # First pad sequences
        padded_sequences = [pad_sequence(seq, max_seq_length) for seq in sequences]

        # Then tokenize
        all_tokens = []
        for seq in padded_sequences:
            tokens = tokenizer.tokenize(
                pd.DataFrame(seq, columns=feature_columns),
                feature_columns
            )
            all_tokens.append(tokens)

        tokens_array = np.array(all_tokens)
        print(f"Tokenized shape: {tokens_array.shape}")

        return EyeTrackingDataset(tokens_array, sequence_labels, sequence_ids,target_positions)

    def custom_collate(batch):
        """
        Custom collate function to handle variable-length target positions.
        Each element in batch is (tokens, label, sequence_id, target_positions)
        """
        # Separate the batch elements
        tokens, labels, sequence_ids, target_positions = zip(*batch)

        # Stack the fixed-size elements
        tokens = torch.stack(tokens)
        labels = torch.stack(labels)

        # Keep variable-length elements as lists
        # sequence_ids and target_positions are already lists, so we can use them as is

        return tokens, labels, list(sequence_ids), list(target_positions)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter.split(np.arange(len(scans_ids)), groups=scans_ids))


    scans_ids_trainval = np.array(scans_ids)[train_val_idx]

    val_size_adjusted = val_size / (1 - test_size)

    val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=seed)
    train_idx_temp, val_idx_temp = next(val_splitter.split(
        np.arange(len(train_val_idx)),
        groups=scans_ids_trainval
    ))
    train_idx = train_val_idx[train_idx_temp]
    val_idx = train_val_idx[val_idx_temp]

    train_groups = set(np.array(scans_ids)[train_idx])
    val_groups = set(np.array(scans_ids)[val_idx])

    print("No overlap between train and val groups:",
          len(train_groups.intersection(val_groups)) == 0)
    # Create dataloaders
    train_groups = set(np.array(scans_ids)[train_idx])
    val_groups = set(np.array(scans_ids)[val_idx])
    test_groups = set(np.array(scans_ids)[test_idx])

    # Print results

    print("Train trial IDs:", train_groups)
    print("Val trial IDs:", val_groups)
    print("Test trial IDs:", test_groups)
    print("No overlap between train and val:", len(train_groups.intersection(val_groups)) == 0)
    print("No overlap between train and test:", len(train_groups.intersection(test_groups)) == 0)
    print("No overlap between val and test:", len(val_groups.intersection(test_groups)) == 0)
    train_dataset = create_dataset(train_idx,scans_data, scans_labels, sequence_ids, tokenizer, feature_columns, max_seq_length,target_positions)
    val_dataset = create_dataset(val_idx,scans_data, scans_labels, sequence_ids, tokenizer, feature_columns, max_seq_length,target_positions)
    test_dataset = create_dataset(test_idx,scans_data, scans_labels, sequence_ids, tokenizer, feature_columns, max_seq_length,target_positions)

    # Print split statistics
    print(f"\nData Split Statistics:")
    print(f"Training sequences: {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    print(f"Test sequences: {len(test_dataset)}")
    print(f"\nLabel Distribution:")
    print(
        f"Train - Positive: {train_dataset.labels.sum()}, Negative: {len(train_dataset.labels) - train_dataset.labels.sum()}")
    print(f"Val - Positive: {val_dataset.labels.sum()}, Negative: {len(val_dataset.labels) - val_dataset.labels.sum()}")
    print(
        f"Test - Positive: {test_dataset.labels.sum()}, Negative: {len(test_dataset.labels) - test_dataset.labels.sum()}")

    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=6,collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=6, collate_fn=custom_collate)

    return train_loader, val_loader, test_loader, tokenizer


def analyze_attention(attentions, target_positions, tolerance=50, top_k=10):
    try:
        num_heads = attentions.shape[0]
        seq_len = attentions.shape[1]

        if seq_len == 0 or num_heads == 0:
            print(f"Invalid attention shape: heads={num_heads}, seq_len={seq_len}")
            return None

        if not target_positions:
            print("No target positions provided")
            return None

        results = {
            "coincidences": 0,
            "total_targets": len(target_positions),
            "total_tokens": seq_len,
            "top_k": top_k,
            "attention_shape": [num_heads, seq_len]
        }

        targets_covered = set()

        for head_index in range(num_heads):
            head_attention = attentions[head_index]
            token_attentions = head_attention.mean(dim=0)

            if torch.isnan(token_attentions).any():
                print(f"NaN values detected in attention scores for head {head_index}")
                continue

            top_k_indices = heapq.nlargest(top_k, range(len(token_attentions)),
                                           token_attentions.cpu().numpy().__getitem__)

            for attention_argmax in top_k_indices:
                for target_pos in target_positions:
                    if abs(attention_argmax - target_pos) <= tolerance and target_pos not in targets_covered:
                        results["coincidences"] += 1
                        targets_covered.add(target_pos)

        return results

    except Exception as e:
        print(f"Error in analyze_attention: {str(e)}")
        return None

def main(method_dir,seed):
    # Load data
    config = DataConfig(
        data_path='data/Categorized_Fixation_Data_1_18.csv',
        approach_num=8,
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
        'CURRENT_FIX_IA_X',
        'CURRENT_FIX_IA_Y',
        'CURRENT_FIX_INDEX',
        'CURRENT_FIX_COMPONENT_COUNT'
    ]
    train_loader, val_loader, test_loader, tokenizer = prepare_data(df, feature_columns,seed=seed)

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(
        vocab_size=tokenizer.vocab_size,
        n_features=len(feature_columns)+2,
        d_model=256,
        pretrained_model='xlnet-base-cased'#"bert-base-uncased"  # or other pre-trained model
    ).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    results_dir = f"transformer_results/{method_dir}"
    os.makedirs(results_dir, exist_ok=True)
    best_model = None
    best_val_acc = 0
    for epoch in range(50):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for tokens, labels,_,_ in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(tokens)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()  # Step every batch instead of every epoch

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for tokens, labels,_,_ in val_loader:
                tokens, labels = tokens.to(device), labels.to(device)
                outputs = model(tokens)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Print progress
        print(f'Epoch {epoch + 1}:')
        print(f'Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Acc: {100. * correct / total:.2f}%')
        print(f'Val Loss: {val_loss / len(val_loader):.4f}, '
              f'Acc: {100. * val_correct / val_total:.2f}%')

        # Early stopping
        if  val_correct / val_total > best_val_acc:
            best_val_acc = val_correct / val_total
            patience_counter = 0
            torch.save(model.state_dict(), f'{results_dir}/best_model.pt')
            best_model  = model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

        scheduler.step()

    model = best_model
    test_loss = 0
    test_correct = 0
    test_total = 0
    true_labels = []
    predicted_labels = []
    all_results = []

    with torch.no_grad():
        total_coincidences = 0
        total_targets = 0
        total_scans_with_targets = 0
        total_valid_positions = 0
        scans_with_no_positions = 0
        scans_with_invalid_positions = 0
        scans_with_null_results = 0

        for tokens, labels, _, target_positions in test_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            outputs, attentions = model(tokens, output_attentions=True)

            # Calculate loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = outputs.max(1)

            # Collect true labels and predictions
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

            for i in range(labels.shape[0]):
                target_positions_in_seq = target_positions[i]
                token_seq = tokens[i]

                if len(target_positions_in_seq) == 0:
                    scans_with_no_positions += 1
                    continue

                max_token_index = token_seq.shape[0] - 1
                valid_target_positions = [pos for pos in target_positions_in_seq if 0 <= pos <= max_token_index]

                if not valid_target_positions:
                    scans_with_invalid_positions += 1
                    continue

                last_layer_attentions = attentions[-1]
                sequence_attentions = last_layer_attentions[i]
                results = analyze_attention(sequence_attentions, valid_target_positions,
                                            tolerance=100, top_k=len(valid_target_positions))

                if results is None:
                    scans_with_null_results += 1
                    continue

                all_results.append(results)
                total_coincidences += results.get("coincidences", 0)
                total_targets += results.get("total_targets", 0)

        print(f"\nDetailed Filtering Summary:")
        print(f"Total scans with targets: {total_scans_with_targets}")
        print(f"Scans with no target positions: {scans_with_no_positions}")
        print(f"Scans with invalid target positions: {scans_with_invalid_positions}")
        print(f"Scans with null results: {scans_with_null_results}")
        print(f"Successfully processed scans: {len(all_results)}")
        print(f"Total valid positions analyzed: {total_valid_positions}")

    # Calculate metrics
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    accuracy = 100. * test_correct / test_total
    test_loss /= len(test_loader)

    # Print test set evaluation results
    print(f'Test Loss: {test_loss:.4f}, '
          f'Acc: {accuracy:.2f}%, '
          f'F1: {f1:.4f}, Recall: {recall:.4f}')

    # Generate classification report
    report = classification_report(true_labels, predicted_labels)
    print('Classification Report:')
    print(report)

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print('Confusion Matrix:')
    print(cm)

    # Save evaluation results
    with open(f'{results_dir}/evaluation_results.txt', 'w') as f:
        f.write(f'Test Loss: {test_loss:.4f}\n')
        f.write(f'Accuracy: {accuracy:.2f}%\n')
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write('\nClassification Report:\n')
        f.write(report)
        f.write('\nConfusion Matrix:\n')
        f.write(str(cm))

    # Save model parameters
    with open(f'{results_dir}/model_parameters.txt', 'w') as f:
        f.write(f'Pre-trained Model: {model.transformer.config._name_or_path}\n')
        f.write(f'Vocab Size: {tokenizer.vocab_size}\n')
        f.write(f'Number of Features: {len(feature_columns)}\n')
        f.write(f'Hidden Size: {model.transformer.config.hidden_size}\n')
        f.write(f'Number of Layers: {model.transformer.config.num_hidden_layers}\n')
        f.write(f'Number of Heads: {model.transformer.config.num_attention_heads}\n')

    interpreter = TransformerInterpreter(
        model=model,
        feature_columns=feature_columns,
        device=device
    )


if __name__ == "__main__":
#best method fine_tune_xlnet-base-cased
# for seed in [0,42, 66, 93]:

    for seed in [42]:
        seed_everything(seed)
        method_dir =f"xlnet _seed_{seed}"
        main(method_dir,seed)