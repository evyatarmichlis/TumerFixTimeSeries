import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# Import your existing modules
from representation_method.utils.general_utils import seed_everything, evaluate_reconstructed_predictions
from representation_method.utils.data_loader import load_eye_tracking_data, DataConfig
from representation_method.utils.data_utils import (
    create_dynamic_time_series, split_train_test_for_time_series,
    create_dynamic_time_series_with_ailment
)
from representation_method.utils.trainers import EnsembleTrainer
from representation_method.models.classifier import CombinedModel, CNN1DModel, GradientLocalizer

FEATURE_COLUMNS = [
    'Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X',
    'CURRENT_FIX_IA_Y', 'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT',
    'WINDOW_MAX_SLICE_FREQ'
]


# ==========================================
# 1. VISUALIZATION & PLOTTING HELPERS
# ==========================================

def plot_participant_panel_of_trial_heatmaps(meta: list[dict], test_df: pd.DataFrame, positive_indices: np.ndarray,
                                             out_dir: str, img_col: str = "CURRENT_FIX_COMPONENT_IMAGE_FILE",
                                             top_k_per_trial: int = 80) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    panel_out = Path(out_dir) / "participant_panels"
    panel_out.mkdir(parents=True, exist_ok=True)

    pred_by_trial = defaultdict(Counter)
    parts_set = set()
    for i in positive_indices:
        if 0 <= i < len(meta):
            m = meta[i]
            trial_key = (m.get("participant_id"), m.get("trial_id"))
            parts_set.add(m.get("participant_id"))
            for s in set(map(str, m.get("window_unique_slices", m.get("window_slices", [])))):
                pred_by_trial[trial_key][s] += 1

    gt_by_trial = defaultdict(set)
    if img_col in test_df.columns:
        for (pid, tid), sub in test_df.groupby(["RECORDING_SESSION_LABEL", "TRIAL_INDEX"]):
            if "target" in sub.columns:
                gt_slices = set(map(str, sub.loc[sub["target"] == 1, img_col].astype(str).unique()))
                gt_by_trial[(pid, tid)] = gt_slices
                parts_set.add(pid)

    participants = sorted([p for p in parts_set if p is not None])
    for pid in participants:
        trials = sorted({tid for (p, tid) in list(pred_by_trial.keys()) + list(gt_by_trial.keys()) if p == pid})
        if not trials: continue

        per_trial_M, per_trial_labels, per_trial_titles, per_trial_widths = [], [], [], []

        for tid in trials:
            trial_key = (pid, tid)
            counter = pred_by_trial.get(trial_key, Counter())
            gt_set = gt_by_trial.get(trial_key, set())
            sub = test_df[(test_df["RECORDING_SESSION_LABEL"] == pid) & (test_df["TRIAL_INDEX"] == tid)]
            ordered_all = list(dict.fromkeys(sub[img_col].astype(str).tolist()))
            all_slices = set(counter.keys()) | set(gt_set)
            ordered = [s for s in ordered_all if s in all_slices]
            if top_k_per_trial: ordered = ordered[:top_k_per_trial]
            n = len(ordered)
            if n == 0: continue

            counts = np.array([counter.get(s, 0) for s in ordered], dtype=float)
            density = counts / counts.max() if counts.max() > 0 else counts
            gt_vec = np.array([1.0 if s in gt_set else 0.0 for s in ordered], dtype=float)
            per_trial_M.append(np.vstack([density[None, :], gt_vec[None, :]]))
            per_trial_labels.append(ordered)
            per_trial_titles.append(f"Trial {tid}")
            per_trial_widths.append(n)

        if not per_trial_M: continue

        n_trials = len(per_trial_M)
        fig, axes = plt.subplots(n_trials, 1,
                                 figsize=(max(12.0, 0.18 * max(per_trial_widths)), max(3.0, 2.2 * n_trials)),
                                 constrained_layout=True)
        if n_trials == 1: axes = [axes]

        for ax, M, labels, title in zip(axes, per_trial_M, per_trial_labels, per_trial_titles):
            n = M.shape[1]
            ax.imshow(M, aspect='auto', interpolation='nearest', vmin=0.0, vmax=1.0, cmap='viridis')
            gt_overlay = np.zeros_like(M)
            gt_overlay[1, :] = M[1, :]
            gt_masked = np.ma.masked_where(gt_overlay == 0, gt_overlay)
            ax.imshow(gt_masked, aspect='auto', interpolation='nearest', vmin=0.0, vmax=1.0,
                      cmap=plt.matplotlib.colors.ListedColormap([[0, 0, 0, 0], [0, 1, 0, 0.85]]))
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Pred", "GT"])
            xticks = np.arange(0, n, max(1, n // 40))
            ax.set_xticks(xticks)
            ax.set_xticklabels([labels[i] for i in xticks], rotation=90, fontsize=8)
            ax.set_title(title, fontsize=11)

        fig.colorbar(axes[0].images[0], ax=axes, fraction=0.02, pad=0.01).set_label('Normalized density', rotation=90)
        fig.suptitle(f"Participant {pid} — Per-trial slice density vs. GT (2 rows per trial)", fontsize=13)
        fig.savefig(panel_out / f"participant_{pid}.png", dpi=150)
        plt.close(fig)


def plot_participant_slice_heatmap_with_gt(meta: list[dict], test_df: pd.DataFrame, positive_indices: np.ndarray,
                                           out_dir: str, img_col: str = "CURRENT_FIX_COMPONENT_IMAGE_FILE",
                                           top_k_per_trial: int = 80) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    part_out = Path(out_dir) / "participant_heatmaps"
    part_out.mkdir(parents=True, exist_ok=True)

    pred_by_trial = defaultdict(Counter)
    parts_set = set()
    for i in positive_indices:
        if 0 <= i < len(meta):
            m = meta[i]
            trial_key = (m.get("participant_id"), m.get("trial_id"))
            parts_set.add(m.get("participant_id"))
            for s in set(map(str, m.get("window_unique_slices", m.get("window_slices", [])))):
                pred_by_trial[trial_key][s] += 1

    gt_by_trial = defaultdict(set)
    if img_col in test_df.columns:
        for (pid, tid), sub in test_df.groupby(["RECORDING_SESSION_LABEL", "TRIAL_INDEX"]):
            if "target" in sub.columns:
                gt_slices = set(map(str, sub.loc[sub["target"] == 1, img_col].astype(str).unique()))
                gt_by_trial[(pid, tid)] = gt_slices
                parts_set.add(pid)

    participants = sorted([p for p in parts_set if p is not None])
    for pid in participants:
        trials = sorted({tid for (p, tid) in list(pred_by_trial.keys()) + list(gt_by_trial.keys()) if p == pid})
        if not trials: continue

        per_trial_density, per_trial_gt, trial_widths, trial_labels = [], [], [], []

        for tid in trials:
            trial_key = (pid, tid)
            counter = pred_by_trial.get(trial_key, Counter())
            gt_set = gt_by_trial.get(trial_key, set())
            sub = test_df[(test_df["RECORDING_SESSION_LABEL"] == pid) & (test_df["TRIAL_INDEX"] == tid)]
            ordered_all = list(dict.fromkeys(sub[img_col].astype(str).tolist()))
            all_slices = set(counter.keys()) | set(gt_set)
            ordered = [s for s in ordered_all if s in all_slices]
            if top_k_per_trial: ordered = ordered[:top_k_per_trial]
            n = len(ordered)
            if n == 0: continue

            counts = np.array([counter.get(s, 0) for s in ordered], dtype=float)
            per_trial_density.append(counts / counts.max() if counts.max() > 0 else counts)
            per_trial_gt.append(np.array([1.0 if s in gt_set else 0.0 for s in ordered], dtype=float))
            trial_widths.append(n)
            trial_labels.append(tid)

        if not per_trial_density: continue

        total_cols = int(np.sum(trial_widths))
        total_rows = 2 * len(per_trial_density)
        M, GT = np.zeros((total_rows, total_cols), dtype=float), np.zeros((total_rows, total_cols), dtype=float)

        col_offsets = np.cumsum([0] + trial_widths[:-1])
        for k, (dens, gtv, w) in enumerate(zip(per_trial_density, per_trial_gt, trial_widths)):
            r0, c0 = 2 * k, int(col_offsets[k])
            M[r0, c0:c0 + w] = dens
            M[r0 + 1, c0:c0 + w] = 0.0
            GT[r0 + 1, c0:c0 + w] = gtv

        fig, ax = plt.subplots(figsize=(max(12.0, 0.03 * total_cols), max(3.0, 0.55 * total_rows)))
        im = ax.imshow(M, aspect='auto', interpolation='nearest', vmin=0.0, vmax=1.0, cmap='viridis')
        ax.imshow(np.ma.masked_where(GT == 0, GT), aspect='auto', interpolation='nearest', vmin=0.0, vmax=1.0,
                  cmap=plt.matplotlib.colors.ListedColormap([[0, 0, 0, 0], [0, 1, 0, 0.85]]))

        yticks, ylabels = [], []
        for k, tid in enumerate(trial_labels):
            yticks.extend([2 * k, 2 * k + 1])
            ylabels.extend([f"Trial {tid} — Pred", f"Trial {tid} — GT"])
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)

        boundaries = list(col_offsets) + [total_cols]
        ax.set_xticks([(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(trial_widths))])
        ax.set_xticklabels([f"T{t}" for t in trial_labels], rotation=0)
        for b in boundaries: ax.axvline(b - 0.5, color='white', linewidth=0.5, alpha=0.6)

        ax.set_title(f"Participant {pid} | Slice prediction density vs. GT (two rows per trial)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Normalized density')
        fig.tight_layout()
        fig.savefig(part_out / f"participant_{pid}.png", dpi=150)
        plt.close(fig)


# ==========================================
# 2. FEATURE ENGINEERING & FILTERING
# ==========================================

def _filter_isolated_windows(predictions: np.ndarray, metadata: list, neighbor_radius: int = 2, min_neighbors: int = 1,
                             group_key: str = "trial_id") -> np.ndarray:
    if predictions.ndim != 1:
        predictions = predictions.ravel()
    pos_idx = np.where(predictions == 1)[0]
    if pos_idx.size == 0:
        return pos_idx

    groups = [m.get(group_key) for m in metadata]
    kept = []
    n = len(predictions)
    for i in pos_idx:
        g = groups[i]
        left = max(0, i - neighbor_radius)
        right = min(n - 1, i + neighbor_radius)
        cnt = sum(1 for j in range(left, right + 1) if j != i and predictions[j] == 1 and groups[j] == g)
        if cnt >= min_neighbors:
            kept.append(i)
    return np.array(kept, dtype=int)


def add_window_max_slice_freq(df: pd.DataFrame, window_size: int, img_col: str = "CURRENT_FIX_COMPONENT_IMAGE_FILE",
                              group_cols=("RECORDING_SESSION_LABEL", "TRIAL_INDEX"),
                              out_col: str = "WINDOW_MAX_SLICE_FREQ") -> pd.DataFrame:
    from collections import deque
    if img_col not in df.columns: raise KeyError(f"Expected '{img_col}'")

    def _rolling_maxfreq(s: pd.Series) -> pd.Series:
        q = deque()
        freq = defaultdict(int)
        out, w = [], max(1, int(window_size))
        for v in s.astype(str).tolist():
            q.append(v)
            freq[v] += 1
            if len(q) > w:
                left = q.popleft()
                freq[left] -= 1
                if freq[left] == 0: del freq[left]
            out.append(max(freq.values()) if freq else 0)
        return pd.Series(out, index=s.index, dtype=float)

    df[out_col] = df.groupby(list(group_cols))[img_col].apply(_rolling_maxfreq).reset_index(
        level=list(range(len(group_cols))), drop=True)
    df[out_col] = df[out_col].fillna(1.0).astype(float)
    return df


def add_image_visit_count(df: pd.DataFrame, img_col: str = "CURRENT_FIX_COMPONENT_IMAGE_FILE",
                          out_col: str = "IMG_VISIT_COUNT",
                          id_cols: tuple[str, ...] = ("RECORDING_SESSION_LABEL", "TRIAL_INDEX")) -> pd.DataFrame:
    run_col = "__img_run_id__"
    df[run_col] = df.groupby(list(id_cols))[img_col].transform(lambda s: (s != s.shift()).cumsum())
    df[out_col] = df.groupby(list(id_cols) + [img_col])[run_col].transform("nunique").astype(float)
    del df[run_col]
    return df


# ==========================================
# 3. CORE PIPELINE RUNNER
# ==========================================

def two_step_pipeline(participant_id, window_size=100, seed=42, remove_isolated: bool = False, neighbor_radius: int = 2,
                      min_neighbors: int = 1):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_columns_train = [
        'Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X',
        'CURRENT_FIX_IA_Y', 'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT'
    ]

    # 1. Load Data
    dir_path = Path(__file__).parent.parent.parent / "fwd_data"
    csv_path = dir_path / 'Combined_Participants_CT_23_12_25.csv'

    config = DataConfig(data_path=csv_path, approach_num=6, normalize=True, per_slice_target=True,
                        participant_id=participant_id)
    df = load_eye_tracking_data(data_path=config.data_path, approach_num=config.approach_num,
                                participant_id=config.participant_id, data_format="legacy")

    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    # 2. Split data
    train_df, test_df = split_train_test_for_time_series(df, test_size=0.2, random_state=seed)
    train_df, val_df = split_train_test_for_time_series(train_df, test_size=0.2, random_state=seed)

    gt_ailments_df = test_df.loc[
        test_df['AILMENT_NUMBER'] != -1, ['RECORDING_SESSION_LABEL', 'TRIAL_INDEX', 'AILMENT_NUMBER']].drop_duplicates()
    total_unique_ailments_in_test = len(gt_ailments_df)

    # 3. Create windows
    X_train, Y_train, train_metadata, _ = create_dynamic_time_series_with_ailment(train_df, feature_columns_train,
                                                                                  window_size=window_size)
    X_val, Y_val, val_metadata, _ = create_dynamic_time_series_with_ailment(val_df, feature_columns_train,
                                                                            window_size=window_size)
    X_test, Y_test, test_metadata, _ = create_dynamic_time_series_with_ailment(test_df, feature_columns_train,
                                                                               window_size=window_size)

    save_dir = f'results/ailment_tracking_participant_{participant_id}'
    os.makedirs(save_dir, exist_ok=True)

    # ==========================================
    # STEP 1: WINDOW-LEVEL PREDICTION
    # ==========================================
    print("\nSTEP 1: Window-Level Prediction")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).permute(0, 2, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).permute(0, 2, 1)

    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_loader = DataLoader(TensorDataset(X_val_tensor, Y_val_tensor), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor), batch_size=32, shuffle=False)

    ensemble_save_path = os.path.join(save_dir, 'ensemble_models')
    os.makedirs(ensemble_save_path, exist_ok=True)

    ensemble_trainer = EnsembleTrainer(
        base_model_class=CNN1DModel,
        model_params={'input_dim': X_train_tensor.shape[1], 'window_size': window_size, 'output_classes': 2},
        n_models=10,
        device=device,
        save_path=ensemble_save_path
    )

    class_counts = np.bincount(Y_train)
    weights = 1.0 / class_counts

    ensemble_trainer.train_ensemble(
        train_dataset=train_dataset,
        val_loader=val_loader,
        batch_size=32,
        epochs=50,
        criterion=nn.CrossEntropyLoss(),
        optimizer_class=optim.Adam,
        optimizer_params={'lr': 0.001},
        majority_weight=weights[0] if len(weights) > 1 else 0.1
    )

    test_predictions = ensemble_trainer.predict(test_loader, minority_weight=1, threshold=0.92, unanimous=False)

    if remove_isolated:
        kept_indices = _filter_isolated_windows(
            predictions=test_predictions,
            metadata=test_metadata,
            neighbor_radius=neighbor_radius,
            min_neighbors=min_neighbors,
            group_key="trial_id",
        )
        new_preds = np.zeros_like(test_predictions)
        new_preds[kept_indices] = 1
        test_predictions = new_preds

    # ==========================================
    # STEP 2: TARGET LOCALIZATION
    # ==========================================
    print("\nSTEP 2: Target Localization within Predicted Positive Windows")
    positive_indices = np.where(test_predictions == 1)[0]

    if len(positive_indices) == 0:
        print("No positive windows for Stage 2.")
        return

    stage1_model_for_loc = ensemble_trainer.models[0]
    localizer = GradientLocalizer(stage1_model_for_loc, device)
    all_stage2_results = []

    for idx in positive_indices:
        window_metadata = test_metadata[idx]
        window_tensor = X_test_tensor[idx].unsqueeze(0)
        predicted_row = localizer.localize_target(window_tensor.clone())
        true_target_positions = window_metadata.get('target_positions', [])

        ailment_found = 'None'
        for detail in window_metadata.get('ailment_details', []):
            if detail['relative_position'] == predicted_row:
                ailment_found = detail['ailment_number']
                break

        all_stage2_results.append({
            'window_index': idx,
            'predicted_row': predicted_row,
            'true_targets': true_target_positions,
            'is_correct': 1 if predicted_row in true_target_positions else 0,
            'ailment_found': ailment_found,
            'ailments_in_window': window_metadata.get('ailment_numbers', [])
        })

    # Output heatmaps
    plot_participant_panel_of_trial_heatmaps(test_metadata, test_df, positive_indices,
                                             out_dir=os.path.join(save_dir, "slice_level"), top_k_per_trial=200)
    plot_participant_slice_heatmap_with_gt(test_metadata, test_df, positive_indices,
                                           out_dir=os.path.join(save_dir, "slice_level"), top_k_per_trial=80)

    # Calculate final baseline accuracy
    results_df = pd.DataFrame(all_stage2_results)
    true_positive_windows_df = results_df[results_df['true_targets'].apply(len) > 0]

    if not true_positive_windows_df.empty:
        print(f"Stage 2 Pinpointing Accuracy (on TP windows): {true_positive_windows_df['is_correct'].mean():.4f}")

    found_ailments = set(results_df[results_df['ailment_found'] != 'None']['ailment_found'])
    print(f"Stage 2 found ailments: {sorted(list(found_ailments))}")
    print(f"Stage 2 found ailments coverage: {len(found_ailments)}/{total_unique_ailments_in_test}")


if __name__ == "__main__":
    for part in range(35, 36):
        print(f"\n{'#' * 20} participant_id {part} {'#' * 20}")
        try:
            two_step_pipeline(
                participant_id=part,
                window_size=10,
                seed=0,
                remove_isolated=True,
                neighbor_radius=1,
                min_neighbors=1
            )
        except Exception as e:
            print(f"Error processing participant {part}: {e}")