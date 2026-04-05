from typing import Dict, List, Tuple
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from pathlib import Path
import matplotlib.pyplot as plt
import json
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from representation_method.models.classifier import GradientLocalizer, CNN1DModel
from representation_method.utils.data_utils import split_train_test_for_time_series
from representation_method.utils.trainers import EnsembleTrainer
from representation_method.utils.general_utils import seed_everything, reconstruct_and_evaluate_efficiency



def compute_trial_level_tolerance_metrics(test_df: pd.DataFrame,
                                          final_df: pd.DataFrame,
                                          tolerance=1,
                                          id_cols=("RECORDING_SESSION_LABEL", "TRIAL_INDEX"),
                                          ia_col="CURRENT_FIX_INTEREST_AREA_LABEL"):
    """
    Computes Trial-Level Confusion Matrix based on Spatial Tolerance.

    Definitions:
    - Trial TP: Flagged for review AND prediction is within tolerance of GT.
    - Trial FP: Flagged for review BUT prediction is NOT within tolerance (or healthy patient).
    - Trial FN: Patient sick, but NOT flagged for review.
    - Trial TN: Patient healthy, and correctly NOT flagged.
    """
    # 1. Identify all unique trials in test set
    all_trials_df = test_df[list(id_cols)].drop_duplicates()
    all_trials = set(map(tuple, all_trials_df.values))

    # 2. Identify Trials that HAVE Ground Truth (Sick Patients)
    gt_rows = test_df[test_df['target'] == 1]
    gt_trials = set(map(tuple, gt_rows[list(id_cols)].drop_duplicates().values))

    # 3. Identify Trials Flagged by Model (Predictions exist)
    if ia_col not in final_df.columns:
        pred_source = test_df.loc[final_df.index]
    else:
        pred_source = final_df
    pred_trials = set(map(tuple, pred_source[list(id_cols)].drop_duplicates().values))

    # 4. Build Spatial Sets for overlapping checks
    # GT Locations per trial
    gt_locs_map = {}
    for keys, grp in gt_rows.groupby(list(id_cols)):
        locs = set(grp[ia_col].dropna().astype(int))
        locs.discard(-1);
        locs.discard(0)
        if locs: gt_locs_map[keys] = locs

    # Pred Locations per trial
    pred_locs_map = {}
    for keys, grp in pred_source.groupby(list(id_cols)):
        locs = set(grp[ia_col].dropna().astype(int))
        locs.discard(-1);
        locs.discard(0)
        if locs: pred_locs_map[keys] = locs

    # 5. Compute Metrics
    TP_trial = 0
    FP_trial = 0
    FN_trial = 0
    TN_trial = 0

    for trial in all_trials:
        is_sick = trial in gt_trials
        is_flagged = trial in pred_trials

        if is_flagged:
            # The doctor is reviewing this trial. Is it accurate?
            if not is_sick:
                # False Alarm: Healthy patient flagged
                FP_trial += 1
            else:
                # Patient is sick. Did we point to the right spot?
                g_set = gt_locs_map.get(trial, set())
                p_set = pred_locs_map.get(trial, set())

                # Check Tolerance Overlap
                hit = False
                for p in p_set:
                    for g in g_set:
                        if abs(p - g) <= tolerance:
                            hit = True
                            break
                    if hit: break

                if hit:
                    TP_trial += 1  # Flagged + Accurate Location
                else:
                    FP_trial += 1  # Flagged + Wrong Location (Spatial FP)
        else:
            # Not Flagged (Auto-Negative)
            if is_sick:
                FN_trial += 1  # Missed Case
            else:
                TN_trial += 1  # Correct Rejection

    total = TP_trial + FP_trial + FN_trial + TN_trial
    precision = TP_trial / (TP_trial + FP_trial) if (TP_trial + FP_trial) else 0.0
    recall = TP_trial / (TP_trial + FN_trial) if (TP_trial + FN_trial) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "Trial_TP": TP_trial,
        "Trial_FP": FP_trial,
        "Trial_FN": FN_trial,
        "Trial_TN": TN_trial,
        "Trial_Precision": precision,
        "Trial_Recall": recall,
        "Trial_F1": f1,
        "Total_Trials": total,
        "Review_Rate": (TP_trial + FP_trial) / total if total else 0.0
    }


# =============================
# NEW: 2D Grid Topology Helper
# =============================
def _get_grid_neighbors(target_ia, rows=5, cols=3):
    """
    Returns a set of IA labels that are within 1 step (Up/Down/Left/Right)
    of the target_ia on a 5x3 grid.
    Includes the target_ia itself.
    Assumes 1-based indexing (1..15).
    """
    try:
        target_ia = int(target_ia)
    except:
        return set()

    if target_ia < 1 or target_ia > rows * cols:
        return set()

    # Convert 1-based ID to 0-based (row, col)
    idx = target_ia - 1
    r, c = divmod(idx, cols)

    neighbors = {target_ia}

    # Up
    if r > 0:
        neighbors.add(target_ia - cols)
    # Down
    if r < rows - 1:
        neighbors.add(target_ia + cols)
    # Left
    if c > 0:
        neighbors.add(target_ia - 1)
    # Right
    if c < cols - 1:
        neighbors.add(target_ia + 1)

    return neighbors


# =============================
# Updated Metrics (Location Level)
# =============================
def compute_tolerance_metrics(test_df: pd.DataFrame,
                              final_df: pd.DataFrame,
                              id_cols=("RECORDING_SESSION_LABEL", "TRIAL_INDEX"),
                              ia_col="CURRENT_FIX_INTEREST_AREA_LABEL",
                              tolerance=1, n_locations=15):
    """
    Computes TP, FP, FN based on 2D Grid Tolerance.
    A prediction is a TP if Pred_IA is a direct neighbor of GT_IA on the 5x3 grid.
    """
    # 1. Get Ground Truth Sets per Trial
    gt_lookup = {}
    gt_rows = test_df[test_df["target"] == 1]
    for keys, group in gt_rows.groupby(list(id_cols)):
        valid_ias = set(group[ia_col].dropna().astype(int).unique())
        valid_ias.discard(-1);
        valid_ias.discard(0)
        if valid_ias:
            gt_lookup[keys] = valid_ias

    # 2. Get Prediction Sets per Trial
    pred_lookup = {}
    if ia_col not in final_df.columns:
        pred_subset = test_df.loc[final_df.index].copy()
    else:
        pred_subset = final_df.copy()

    for keys, group in pred_subset.groupby(list(id_cols)):
        valid_ias = set(group[ia_col].dropna().astype(int).unique())
        valid_ias.discard(-1);
        valid_ias.discard(0)
        if valid_ias:
            pred_lookup[keys] = valid_ias

    # 3. Compute Metrics
    TP = 0
    FP = 0
    FN = 0
    all_keys = set(gt_lookup.keys()) | set(pred_lookup.keys())

    for key in all_keys:
        gt_set = gt_lookup.get(key, set())
        pred_set = pred_lookup.get(key, set())

        # --- Check Predictions (TP / FP) ---
        for p_ia in pred_set:
            hit = False
            for g_ia in gt_set:
                # Use 2D Neighbor check instead of linear abs()
                if tolerance > 0:
                    valid_neighbors = _get_grid_neighbors(g_ia)
                    if p_ia in valid_neighbors:
                        hit = True
                        break
                else:
                    # Exact match
                    if p_ia == g_ia:
                        hit = True
                        break
            if hit:
                TP += 1
            else:
                FP += 1

        # --- Check Ground Truths (FN) ---
        for g_ia in gt_set:
            covered = False
            # Define the "Win Zone" for this GT
            if tolerance > 0:
                win_zone = _get_grid_neighbors(g_ia)
            else:
                win_zone = {g_ia}

            # Did any prediction fall into the win zone?
            if not pred_set.isdisjoint(win_zone):
                covered = True

            if not covered:
                FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    total_slots = len(all_keys) * n_locations
    TN = max(0, total_slots - (TP + FP + FN))
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    cm = np.array([[TN, FP], [FN, TP]])
    print("Location Level CM (2D Tol):")
    print(cm)

    return {
        "TP_Tolerance": TP, "FP_Tolerance": FP, "FN_Tolerance": FN,
        "Precision_Tolerance": precision, "Recall_Tolerance": recall, "F1_Tolerance": f1,
        "Tolerance_Value": tolerance, "CM_Tolerance": cm
    }



# =============================
# Updated Visualization (2D Neighbors)
# =============================
def plot_ekg_spatial_stripes_panel(test_metadata: List[dict],
                                   test_df: pd.DataFrame,
                                   positive_indices: np.ndarray,
                                   out_dir: str,
                                   participant_id: str,
                                   ia_col: str = "CURRENT_FIX_INTEREST_AREA_LABEL",
                                   tolerance=1):
    """
    Generates stacked panel.
    GT Row includes 2D GRID DILATION (Light Green) to visualize tolerance.
    """
    data_map = _build_spatial_stripes_data(
        test_metadata, test_df, positive_indices, participant_id, ia_col
    )

    trials = sorted(data_map.keys())
    if not trials: return

    n_trials = len(trials)
    fig_h = max(4.0, 0.6 * n_trials)
    fig, axes = plt.subplots(n_trials, 1, figsize=(10, fig_h), constrained_layout=True)
    if n_trials == 1: axes = [axes]

    TARGET_LABELS = [str(i) for i in range(1, 16)]

    for i, tid in enumerate(trials):
        ax = axes[i]
        gt_vec, pred_vec = data_map[tid]

        # --- Apply 2D Dilation to GT for Visualization ---
        gt_dilated = gt_vec.copy()
        indices = np.where(gt_vec > 0)[0]  # 0-based indices (0..14)

        for idx in indices:
            ia_label = idx + 1  # Convert to 1-based label
            # Get neighbors on 5x3 grid
            neighbors = _get_grid_neighbors(ia_label) if tolerance > 0 else {ia_label}

            for neighbor in neighbors:
                n_idx = neighbor - 1  # Back to 0-based
                # If neighbor spot is empty in the plot, paint it Light Green (0.4)
                # We don't overwrite the Dark Green (1.0)
                if gt_dilated[n_idx] == 0:
                    gt_dilated[n_idx] = 0.4

        # Stack: Row 0 is GT (Dilated), Row 1 is Pred
        stack = np.vstack([gt_dilated, pred_vec])

        # Plot GT Layer (Greens)
        gt_layer = np.zeros_like(stack)
        gt_layer[0, :] = gt_dilated
        mask_gt = np.ones_like(stack, dtype=bool)
        mask_gt[0, :] = (gt_dilated == 0)

        ax.imshow(np.ma.masked_array(gt_layer, mask_gt),
                  cmap='Greens', vmin=0, vmax=1, aspect='auto', interpolation='nearest')

        # Plot Pred Layer (Reds)
        pred_layer = np.zeros_like(stack)
        pred_layer[1, :] = pred_vec
        mask_pred = np.ones_like(stack, dtype=bool)
        mask_pred[1, :] = False

        ax.imshow(np.ma.masked_array(pred_layer, mask_pred),
                  cmap='Reds', vmin=0, vmax=1, aspect='auto', interpolation='nearest')

        # Styling
        ax.set_xticks(np.arange(15))
        ax.set_xticks(np.arange(15) - 0.5, minor=True)
        ax.set_yticks([0, 1])
        ax.set_yticks([0.5], minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', bottom=False, left=False)
        ax.set_yticklabels(["GT", "Pred"], fontsize=9, fontweight='bold')

        if i == n_trials - 1:
            ax.set_xticklabels(TARGET_LABELS)
            ax.set_xlabel(f"Interest Area (Location) - Light Green = 2D Neighbor")
        else:
            ax.set_xticklabels([])

        ax.set_ylabel(f"Trial {tid}", rotation=0, ha='right', va='center', fontsize=9)

    fig.suptitle(f"Participant {participant_id}: Spatial Accuracy (2D Neighbor Tolerance)", y=1.01, fontsize=14)

    out_base = Path(out_dir) / "ekg_spatial_strips"
    out_base.mkdir(parents=True, exist_ok=True)
    out_png = out_base / f"spatial_strip_P{participant_id}.png"
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved per-trial spatial strip panel -> {out_png}")


# EKG Heatmap (Time-series density) Helpers
# =============================
def _build_spatial_stripes_data(test_metadata: List[dict],
                                test_df: pd.DataFrame,
                                positive_indices: np.ndarray,
                                participant_id: str,
                                ia_col="CURRENT_FIX_INTEREST_AREA_LABEL"):
    """
    Builds normalized 1x15 vectors for GT and Prediction for every trial.
    Target IAs: [1, 2, ... 15]
    """
    # 1. Define Target Locations (Columns)
    TARGET_IAS = np.arange(1, 16)  # [1..15]
    ia_to_idx = {ia: i for i, ia in enumerate(TARGET_IAS)}
    n_cols = len(TARGET_IAS)

    pid_str = str(participant_id)
    # Filter DF for this participant
    sub_df = test_df[test_df["RECORDING_SESSION_LABEL"].astype(str) == pid_str].copy()
    if sub_df.empty:
        return {}

    # 2. Pre-process Positive Windows for fast lookup
    # Map: trial_id -> list of (start_idx, end_idx)
    pred_windows_lookup = {}

    # Filter positive indices belonging to this participant
    part_indices = [i for i in positive_indices
                    if i < len(test_metadata)
                    and str(test_metadata[i].get("participant_id")) == pid_str]

    for idx in part_indices:
        m = test_metadata[idx]
        tid = m.get("trial_id")
        s = m.get("window_start_idx")
        e = m.get("window_end_idx")
        if tid not in pred_windows_lookup:
            pred_windows_lookup[tid] = []
        pred_windows_lookup[tid].append((s, e))

    # 3. Build Vectors per Trial
    trial_results = {}
    grouped = sub_df.groupby("TRIAL_INDEX")

    for tid, trial_data in grouped:
        # Init 1D vectors
        gt_vec = np.zeros(n_cols, dtype=float)
        pred_vec = np.zeros(n_cols, dtype=float)

        # --- Fill GT Vector ---
        # Find all IAs where target == 1
        gt_hits = trial_data[trial_data["target"] == 1][ia_col].unique()
        for ia in gt_hits:
            if ia in ia_to_idx:
                gt_vec[ia_to_idx[ia]] = 1.0

        # --- Fill Pred Vector ---
        # Find all IAs visited during positive windows
        if tid in pred_windows_lookup:
            # We need to access rows by integer position relative to the trial start
            # Assuming trial_data is sorted and contiguous
            trial_data = trial_data.reset_index(drop=True)  # Ensure 0..N indexing for iloc

            for (w_start, w_end) in pred_windows_lookup[tid]:
                # Safe slicing
                valid_s = max(0, w_start)
                valid_e = min(len(trial_data), w_end)

                if valid_e > valid_s:
                    # Get IAs visited in this window
                    window_ias = trial_data.iloc[valid_s:valid_e][ia_col].unique()
                    for ia in window_ias:
                        if ia in ia_to_idx:
                            # Accumulate "suspicion" for this location
                            pred_vec[ia_to_idx[ia]] += 1.0

        # Normalize Pred Vector (Local density)
        if pred_vec.max() > 0:
            pred_vec /= pred_vec.max()

        trial_results[tid] = (gt_vec, pred_vec)

    return trial_results


def _build_ekg_trial_density_and_gt(test_metadata: List[dict],
                                    test_df: pd.DataFrame,
                                    positive_indices: np.ndarray,
                                    id_cols=("RECORDING_SESSION_LABEL", "TRIAL_INDEX"),
                                    index_col="CURRENT_FIX_INDEX"):
    """
    Build per-trial density of Step-1 positive windows over the EKG time index axis,
    alongside the ground-truth target mask.

    Density[i] counts how many predicted-positive windows cover time-step i.
    We then min-max normalize density per trial to [0,1] for plotting.

    Returns dict[(participant, trial)] -> (indices, density_norm, gt_mask)
    """
    # Map each positive window to its (participant, trial) and its covered index range
    pos_meta = [test_metadata[i] for i in positive_indices] if len(positive_indices) else []

    # Group trial lengths and GT by (participant, trial)
    trial_lookup = {}
    # Ensure we iterate through known trials in test_df
    grouped = test_df.groupby(list(id_cols))

    for (pid, tid), trial_df in grouped:
        # Sort by index to ensure time alignment
        trial_df = trial_df.sort_values(by=index_col)
        idx = trial_df[index_col].astype(int).to_numpy()

        # Determine the span of the trial (assuming 0-indexed or continuous)
        if len(idx) == 0:
            continue

        # We map indices to a dense array from 0 to max_index
        # (or just use relative position if indices are large timestamps)
        # Here we assume CURRENT_FIX_INDEX is a relative step counter starting near 0.
        max_idx = int(idx.max())
        n = max_idx + 1

        gt_mask = np.zeros(n, dtype=int)
        # Fill GT where target is 1
        # Note: This assumes idx values directly map to array indices.
        # If CURRENT_FIX_INDEX is arbitrary, we might need a mapping,
        # but usually in these pipelines it's 0,1,2...
        targets = trial_df["target"].astype(int).to_numpy()
        gt_mask[idx] = targets

        # init density array
        dens = np.zeros(n, dtype=float)

        trial_lookup[(pid, tid)] = {
            "n": n,
            "gt": gt_mask,
            "dens": dens,
        }

    # Accumulate density from positive windows
    for m in pos_meta:
        pid = m.get("participant_id")
        tid = m.get("trial_id")
        start_i = int(m.get("window_start_idx", 0))
        end_i = int(m.get("window_end_idx", 0))

        key = (pid, tid)
        if key not in trial_lookup:
            continue

        n = trial_lookup[key]["n"]
        # Clip window bounds to trial length
        l = max(0, min(start_i, n))
        r = max(0, min(end_i, n))

        if r > l:
            # Increment density for all time steps covered by this window
            trial_lookup[key]["dens"][l:r] += 1.0

    # Normalize density per trial to [0,1]
    out = {}
    for key, rec in trial_lookup.items():
        dens = rec["dens"]
        mx = float(dens.max()) if dens.size else 0.0
        dens_norm = dens / mx if mx > 0 else dens
        out[key] = (np.arange(rec["n"]), dens_norm, rec["gt"])  # (axis, density, gt)
    return out



def export_row_prediction_table(
        test_df: pd.DataFrame,
        final_df_majority: pd.DataFrame,
        final_df_any: pd.DataFrame,
        out_csv: Path | str,
        id_cols=("RECORDING_SESSION_LABEL", "TRIAL_INDEX"),
        slice_col: str = "CURRENT_FIX_COMPONENT_IMAGE_FILE",
        ailment_col: str = "AILMENT_NUMBER",
        loc_col: str = "CURRENT_FIX_INTEREST_AREA_LABEL",
) -> pd.DataFrame:
    """
    Return & save a per-row table with GT and predicted labels for Majority/Any.
    """
    base_cols = list(id_cols) + ["CURRENT_FIX_INDEX", "target"]
    have_slice = slice_col in test_df.columns
    if have_slice:
        base_cols.append(slice_col)
    have_ail = ailment_col in test_df.columns
    if have_ail:
        base_cols.append(ailment_col)
    have_loc = loc_col in test_df.columns
    if have_loc:
        base_cols.append(loc_col)

    df_out = test_df[base_cols].copy()
    # Rename for friendliness
    rename_map = {"CURRENT_FIX_INDEX": "row_index"}
    if have_slice:
        rename_map[slice_col] = "slice_image"
    if have_ail:
        rename_map[ailment_col] = "ailment_name"
    if have_loc:
        rename_map[loc_col] = "interest_area_label"
    df_out.rename(columns=rename_map, inplace=True)

    # Initialize predictions
    df_out["pred_majority"] = 0
    df_out["pred_any"] = 0

    # Mark Majority predictions using index alignment
    if len(final_df_majority) > 0:
        idx_m = test_df.index.get_indexer(final_df_majority.index)
        idx_m = np.unique(idx_m[idx_m >= 0])
        if idx_m.size:
            df_out.loc[df_out.index[idx_m], "pred_majority"] = 1

    # Mark Any predictions using index alignment
    if len(final_df_any) > 0:
        idx_a = test_df.index.get_indexer(final_df_any.index)
        idx_a = np.unique(idx_a[idx_a >= 0])
        if idx_a.size:
            df_out.loc[df_out.index[idx_a], "pred_any"] = 1

    # Make ailment more readable: empty string for background
    if "ailment_name" in df_out.columns:
        def _fmt_ail(x):
            try:
                xi = int(x)
                return "" if xi == -1 else str(xi)
            except Exception:
                return "" if pd.isna(x) else str(x)

        df_out["ailment_name"] = df_out["ailment_name"].apply(_fmt_ail)
    else:
        df_out["ailment_name"] = ""

    if "interest_area_label" not in df_out.columns:
        df_out["interest_area_label"] = ""

    # Save
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"Saved row-level prediction table -> {out_csv}")
    return df_out


# =============================
# Split + Metrics Helper Utils
# =============================
def _to_jsonable(obj):
    import numpy as _np
    import pandas as _pd
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, (set,)):
        return sorted(list(obj))
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if isinstance(obj, (dict,)):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (_pd.Series,)):
        return obj.to_dict()
    if isinstance(obj, (_pd.DataFrame,)):
        return obj.to_dict(orient='records')
    return str(obj)


def build_summary_dataframe(suite_results: dict) -> pd.DataFrame:
    """Flatten nested results into a comparison table."""
    rows = []
    for scope, experiments in suite_results.items():
        if not isinstance(experiments, dict):
            continue
        for exp_key, res in experiments.items():
            if not isinstance(res, dict):
                continue
            for strategy, metrics in res.items():
                if not isinstance(metrics, dict):
                    continue
                n_pred_rows = metrics.get("n_pred_rows")

                # Window-level
                if "window" in metrics:
                    m = metrics["window"]
                    rows.append({
                        "scope": scope, "experiment": exp_key, "strategy": strategy, "level": "window",
                        "precision": m.get("precision"), "recall": m.get("recall"), "f1": m.get("f1"),
                        "TP": m.get("TP"), "FP": m.get("FP"), "TN": m.get("TN"), "FN": m.get("FN"),
                        "TP_A": None, "FP_A": None, "A_size": None, "n_pred_rows": n_pred_rows,
                    })
                # Row-level
                if "row" in metrics:
                    m = metrics["row"]
                    rows.append({
                        "scope": scope, "experiment": exp_key, "strategy": strategy, "level": "row",
                        "precision": m.get("precision"), "recall": m.get("recall"), "f1": m.get("f1"),
                        "TP": m.get("TP"), "FP": m.get("FP"), "TN": m.get("TN"), "FN": m.get("FN"),
                        "TP_A": None, "FP_A": None, "A_size": None, "n_pred_rows": n_pred_rows,
                    })
                # Ailment-level
                if "ailment" in metrics:
                    m = metrics["ailment"]
                    rows.append({
                        "scope": scope, "experiment": exp_key, "strategy": strategy, "level": "ailment",
                        "precision": m.get("precision"), "recall": m.get("recall"), "f1": m.get("f1"),
                        "TP": None, "FP": None, "TN": None, "FN": None,
                        "TP_A": m.get("TP_A"), "FP_A": m.get("FP_A"), "A_size": m.get("A_size"),
                        "n_pred_rows": n_pred_rows,
                    })
    return pd.DataFrame(rows)


def save_suite_results(suite_results: dict, base_dir: Path = Path("results/ekg_experiments")):
    base_dir.mkdir(parents=True, exist_ok=True)
    out_json = base_dir / "summary.json"
    with open(out_json, "w") as f:
        json.dump(suite_results, f, default=_to_jsonable, indent=2)
    df = build_summary_dataframe(suite_results)
    out_csv = base_dir / "summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved summary JSON -> {out_json}")
    print(f"Saved summary CSV  -> {out_csv}")


def _window_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}


def _row_confusion(test_df: pd.DataFrame, final_df: pd.DataFrame) -> Dict[str, int]:
    y_true = test_df["target"].astype(int).values
    y_pred = np.zeros(len(test_df), dtype=int)
    if len(final_df):
        idxer = test_df.index.get_indexer(final_df.index)
        idxer = np.unique(idxer[idxer >= 0])
        if idxer.size:
            y_pred[idxer] = 1
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}


def _ailment_metrics(final_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
    # Precision via ailment_number mode
    prec = compute_ailment_precision(
        final_df, test_df,
        id_cols=("RECORDING_SESSION_LABEL", "TRIAL_INDEX"),
        loc_col="CURRENT_FIX_INTEREST_AREA_LABEL",
        gt_mode="ailment_number"
    )
    # Recall (coverage by tuples)
    gt_ailments_df = (
        test_df.loc[test_df['AILMENT_NUMBER'] != -1,
        ['RECORDING_SESSION_LABEL', 'TRIAL_INDEX', 'AILMENT_NUMBER']]
        .drop_duplicates()
    )
    total_unique = len(gt_ailments_df)
    found = len(extract_ailment_tuples_from_predictions(
        final_df, test_df,
        id_cols=("RECORDING_SESSION_LABEL", "TRIAL_INDEX"),
        ailment_col="AILMENT_NUMBER", na_value=-1
    ))
    recall = (found / total_unique) if total_unique else 0.0
    precision = float(prec.get("precision_A", 0.0))
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "TP_A": float(prec.get("TP_A", 0)), "FP_A": float(prec.get("FP_A", 0)),
        "A_size": float(prec.get("A_size", total_unique)),
    }


def _make_val_split(train_df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42, by_trials: bool = True):
    if by_trials:
        keys_cols = ['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']
        keys = train_df[keys_cols].drop_duplicates()
        n = len(keys)
        n_val = max(1, int(round(val_ratio * n))) if n > 1 else 1
        val_sel = keys.sample(n=n_val, random_state=seed)
        left = train_df.reset_index()
        val_idx = left.merge(val_sel, on=keys_cols)['index'].to_numpy()
        val_df = train_df.loc[val_idx]
        tr_df = train_df.drop(index=val_idx)
        return tr_df, val_df
    else:
        val_df = train_df.sample(frac=val_ratio, random_state=seed)
        tr_df = train_df.drop(index=val_df.index)
        return tr_df, val_df


def _safe_div(n, d):
    return float(n) / d if d else 0.0


def compute_ailment_precision(
        final_df: pd.DataFrame, test_df: pd.DataFrame,
        id_cols=("RECORDING_SESSION_LABEL", "TRIAL_INDEX"),
        loc_col="CURRENT_FIX_INTEREST_AREA_LABEL",
        gt_mode="ailment_number", ailment_number_na_value=-1
):
    if isinstance(id_cols, str): id_cols = [id_cols]
    if gt_mode == "ailment_number":
        if "AILMENT_NUMBER" not in test_df.columns:
            raise KeyError("gt_mode='ailment_number' requires 'AILMENT_NUMBER' in test_df")

        A_df = (
            test_df.loc[test_df["AILMENT_NUMBER"] != ailment_number_na_value, list(id_cols) + ["AILMENT_NUMBER"]]
            .dropna(subset=["AILMENT_NUMBER"]).drop_duplicates()
        )
        A = set(map(tuple, A_df.values))

        pred_cols_needed = list(id_cols) + [loc_col, "AILMENT_NUMBER"]
        if all(c in final_df.columns for c in pred_cols_needed):
            pred_source = final_df[pred_cols_needed]
        else:
            pred_source = test_df.loc[final_df.index, pred_cols_needed]

        Lhat_ail_df = (
            pred_source.loc[
                pred_source["AILMENT_NUMBER"] != ailment_number_na_value, list(id_cols) + ["AILMENT_NUMBER"]]
            .drop_duplicates()
        )
        Lhat_ail = set(map(tuple, Lhat_ail_df.values))

        FP_loc_df = (
            pred_source.loc[pred_source["AILMENT_NUMBER"] == ailment_number_na_value, list(id_cols) + [loc_col]]
            .dropna(subset=[loc_col]).drop_duplicates()
        )

        tp = len(A & Lhat_ail)
        fp = len(FP_loc_df)
        precision = _safe_div(tp, tp + fp)
        return {"A_size": len(A), "TP_A": tp, "FP_A": fp, "precision_A": precision}
    else:
        # Legacy target based logic
        gt_mask = (test_df["target"] == 1)
        A_df = test_df.loc[gt_mask, list(id_cols) + [loc_col]].dropna(subset=[loc_col]).drop_duplicates()
        A = set(map(tuple, A_df.values))

        if all(c in final_df.columns for c in list(id_cols) + [loc_col]):
            pred_source = final_df[list(id_cols) + [loc_col]]
        else:
            pred_source = test_df.loc[final_df.index, list(id_cols) + [loc_col]]

        Lhat_df = pred_source.dropna(subset=[loc_col]).drop_duplicates()
        Lhat = set(map(tuple, Lhat_df.values))
        tp = len(A & Lhat)
        fp = len(Lhat - A)
        precision = _safe_div(tp, tp + fp)
        return {"A_size": len(A), "TP_A": tp, "FP_A": fp, "precision_A": precision}


def extract_ailment_tuples_from_predictions(
        predicted_df: pd.DataFrame, test_df: pd.DataFrame,
        id_cols=("RECORDING_SESSION_LABEL", "TRIAL_INDEX"),
        ailment_col="AILMENT_NUMBER", na_value=-1,
):
    id_cols = list(id_cols)
    cols = id_cols + [ailment_col]
    if all(c in predicted_df.columns for c in cols):
        src = predicted_df[cols]
    else:
        src = test_df.loc[predicted_df.index, cols]
    sub = src.loc[src[ailment_col] != na_value].drop_duplicates()
    return set(map(tuple, sub.values))


def simulate_random_search_and_compare(pipeline_df, test_df, strategy_name, total_unique_ailments, n_simulations=1,
                                       id_cols=("RECORDING_SESSION_LABEL", "TRIAL_INDEX"),
                                       ailment_col="AILMENT_NUMBER", na_value=-1):
    k = len(pipeline_df)
    if k == 0:
        print(f"Strategy '{strategy_name}' selected 0 rows.")
        return
    pipeline_ailments = extract_ailment_tuples_from_predictions(
        pipeline_df, test_df, id_cols=id_cols, ailment_col=ailment_col, na_value=na_value
    )
    random_ailments_found = []
    for _ in range(n_simulations):
        random_sample = test_df.sample(n=min(k, len(test_df)), replace=False)
        uniq = (
            random_sample.loc[random_sample[ailment_col] != na_value, list(id_cols) + [ailment_col]]
            .drop_duplicates()
        )
        random_ailments_found.append(len(uniq))
    avg_random = float(np.mean(random_ailments_found)) if random_ailments_found else 0.0
    print(f"'{strategy_name}' found: {len(pipeline_ailments)}/{total_unique_ailments}")
    print(f"Random found (avg): {avg_random:.2f}/{total_unique_ailments}")
    if avg_random > 0:
        print(f"Lift: {len(pipeline_ailments) / avg_random:.2f}x")


def create_dynamic_time_series_for_ekg_with_ailment(df, feature_columns, window_size=100, window_step=1):
    windows, labels, metadata, ailment_locations = [], [], [], []
    for (participant_id, trial_id), trial_df in df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']):
        trial_length = len(trial_df)
        trial_df = trial_df.reset_index(drop=True)
        for start_idx in range(0, trial_length - window_size + 1, window_step):
            end_idx = start_idx + window_size
            window = trial_df.iloc[start_idx:end_idx]
            if len(window) < window_size: continue
            target_array = window['target'].values
            window_target_positions = np.where(target_array == 1)[0]
            window_features = window[feature_columns].values

            ailment_numbers = window['AILMENT_NUMBER'].values if 'AILMENT_NUMBER' in window.columns else np.full(
                window_size, -1)
            window_ailment_positions = np.where(ailment_numbers != -1)[0]
            unique_ailments = list(set(ailment_numbers[ailment_numbers != -1]))

            windows.append(window_features)
            labels.append(int(len(window_target_positions) > 0))
            ailment_locations.append(window_ailment_positions)
            window_meta = {
                'participant_id': participant_id, 'trial_id': trial_id,
                'window_start_idx': start_idx, 'window_end_idx': end_idx,
                'target_positions': window_target_positions.tolist(),
                'has_target': int(len(window_target_positions) > 0),
                'ailment_numbers': unique_ailments,
                'has_valid_ailment': len(unique_ailments) > 0,
                'ailment_details': [{'relative_position': pos, 'ailment_number': str(ailment)}
                                    for pos, ailment in enumerate(ailment_numbers) if ailment != -1]
            }
            metadata.append(window_meta)
    return np.array(windows), np.array(labels), metadata, np.array(ailment_locations, dtype=object)


# =============================
# Runner: EKG Pipeline on Splits
# =============================

def run_ekg_pipeline_on_splits(train_df: pd.DataFrame,
                               val_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               window_size: int = 50,
                               seed: int = 42,
                               window_step: int = 1,
                               save_dir: str = "results/ekg_experiments") -> Dict:
    """Run the existing EKG two-step pipeline given explicit splits."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)

    feature_columns = [
        'Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_INDEX',
        'CURRENT_FIX_COMPONENT_COUNT', 'rolling_mean_10', 'rolling_std_10', 'signal_derivative'
    ]

    X_train, Y_train, train_meta, _ = create_dynamic_time_series_for_ekg_with_ailment(
        train_df, feature_columns, window_size=window_size, window_step=window_step
    )
    X_val, Y_val, val_meta, _ = create_dynamic_time_series_for_ekg_with_ailment(
        val_df, feature_columns, window_size=window_size, window_step=window_step
    )
    X_test, Y_test, test_meta, _ = create_dynamic_time_series_for_ekg_with_ailment(
        test_df, feature_columns, window_size=window_size, window_step=window_step
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).permute(0, 2, 1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).permute(0, 2, 1)
    Y_train_t = torch.tensor(Y_train, dtype=torch.long)
    Y_val_t = torch.tensor(Y_val, dtype=torch.long)
    Y_test_t = torch.tensor(Y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, Y_train_t)
    val_loader = DataLoader(TensorDataset(X_val_t, Y_val_t), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t), batch_size=32, shuffle=False)

    en = EnsembleTrainer(
        base_model_class=CNN1DModel,
        model_params={'input_dim': X_train_t.shape[1], 'window_size': window_size, 'output_classes': 2},
        n_models=10, device=device, save_path=os.path.join(save_dir, 'ensemble_models')
    )
    class_counts = np.bincount(Y_train)
    weights = 1.0 / class_counts if class_counts.sum() else np.array([1.0, 1.0])

    en.train_ensemble(
        train_dataset=train_ds,
        val_loader=val_loader,
        batch_size=32,
        epochs=100,
        criterion=nn.CrossEntropyLoss(),
        optimizer_class=optim.Adam,
        optimizer_params={'lr': 0.001},
        majority_weight=weights[0] if len(weights) > 1 else 1.0
    )

    best_threshold = en.find_best_threshold(val_loader)
    y_pred_w = en.predict(test_loader, threshold=best_threshold)

    win_conf = _window_confusion(Y_test, y_pred_w)
    win_prec = precision_score(Y_test, y_pred_w, zero_division=0)
    win_rec = recall_score(Y_test, y_pred_w, zero_division=0)
    win_f1 = f1_score(Y_test, y_pred_w, zero_division=0)

    pos_idx = np.where(y_pred_w == 1)[0]

    # === NEW: EKG cell-level density heatmaps per participant (using Step-1 positives) ===
    try:
        test_participants = (
            test_df["RECORDING_SESSION_LABEL"].dropna().astype(str).unique().tolist()
            if "RECORDING_SESSION_LABEL" in test_df.columns else []
        )
        for pid in test_participants:
            plot_ekg_spatial_stripes_panel(
                test_metadata=test_meta,
                test_df=test_df,
                positive_indices=pos_idx,
                out_dir=save_dir,
                participant_id=str(pid),
                ia_col="CURRENT_FIX_INTEREST_AREA_LABEL"
            )
    except Exception as e:
        print(f"[WARN] Failed to plot EKG density heatmaps (run_ekg_pipeline_on_splits): {e}")

    all_stage2_results = []
    if len(pos_idx):
        loc = GradientLocalizer(en.models[0], device)
        for idx in pos_idx:
            meta = test_meta[idx]
            w_t = X_test_t[idx].unsqueeze(0)
            pred_row = loc.localize_target(w_t.clone())
            y_true_rows = meta.get('target_positions', [])
            is_correct = 1 if pred_row in y_true_rows else 0
            ailment_found = 'None'
            for detail in meta.get('ailment_details', []):
                if detail['relative_position'] == pred_row:
                    ailment_found = detail['ailment_number']
                    break
            all_stage2_results.append({
                'window_index': idx,
                'predicted_row': pred_row,
                'true_targets': y_true_rows,
                'is_correct': is_correct,
                'ailment_found': ailment_found,
                'ailments_in_window': meta.get('ailment_numbers', [])
            })

    results = {}
    for strategy in ("majority", "any", "all"):
        if len(all_stage2_results):
            final_df, eff = reconstruct_and_evaluate_efficiency(all_stage2_results, test_df, test_meta,
                                                                strategy=strategy)
        else:
            final_df = pd.DataFrame(columns=test_df.columns).iloc[:0]
            eff = {}

        row_conf = _row_confusion(test_df, final_df)
        row_prec = _safe_div(row_conf["TP"], (row_conf["TP"] + row_conf["FP"]))
        row_rec = _safe_div(row_conf["TP"], (row_conf["TP"] + row_conf["FN"]))
        row_f1 = (2 * row_prec * row_rec / (row_prec + row_rec)) if (row_prec + row_rec) else 0.0
        ail = _ailment_metrics(final_df, test_df)

        results[strategy] = {
            "window": {"precision": win_prec, "recall": win_rec, "f1": win_f1, **win_conf},
            "row": {"precision": row_prec, "recall": row_rec, "f1": row_f1, **row_conf},
            "ailment": ail,
            "efficiency": eff,
            "n_pred_rows": int(len(final_df)),
        }

    return results


# =============================
# Main Pipeline for EKG Data
# =============================
def ekg_two_step_pipeline(participant_id, window_size=50, seed=42, window_step=1):
    """
    Two-step pipeline adapted for EKG data with AILMENT_NUMBER tracking.
    Handles participant_id=None by plotting heatmaps for ALL participants in the test set.
    """
    print(f"Running EKG Two-Step Pipeline for Participant {participant_id}")
    print(f"Window size: {window_size}, Seed: {seed}")
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    feature_columns = [
        'Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_INDEX',
        'CURRENT_FIX_COMPONENT_COUNT', 'rolling_mean_10', 'CURRENT_FIX_INTEREST_AREA_LABEL',
        'rolling_std_10', 'signal_derivative'
    ]
    input_columns = ['CURRENT_FIX_INDEX', 'Pupil_Size', 'CURRENT_FIX_DURATION', 'AILMENT_NUMBER',
                     'rolling_mean_10', 'rolling_std_10', 'signal_derivative',
                     'CURRENT_FIX_INTEREST_AREA_LABEL']

    # 1. Load and preprocess EKG data
    csv_path = Path(__file__).parent.parent / "EKG data" / "ML_ECG_Data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found at: {csv_path}")

    df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
    df['target'] = np.where(df['LOCATION_TYPE'] == 'MI_HIT', 1, 0)

    if 'AILMENT_NUMBER' in df.columns:
        df['AILMENT_NUMBER'] = pd.to_numeric(df['AILMENT_NUMBER'], errors='coerce').fillna(-1).astype(int)
    else:
        df['AILMENT_NUMBER'] = -1




    # If participant_id is provided, filter upfront. If None, use all data.
    if participant_id is not None:
        # Ensure type match
        if isinstance(df['RECORDING_SESSION_LABEL'].iloc[0], str):
            pid_query = str(participant_id)
        else:
            try:
                pid_query = int(participant_id)
            except:
                pid_query = participant_id

        df = df[df['RECORDING_SESSION_LABEL'] == pid_query]

    if df.empty:
        print(f"No data found for participant {participant_id}. Exiting.")
        return

    SIGNAL_COL = 'Pupil_Size'
    df['rolling_mean_10'] = df[SIGNAL_COL].rolling(window=10, min_periods=1).mean()
    df['rolling_std_10'] = df[SIGNAL_COL].rolling(window=10, min_periods=1).std()
    df['signal_derivative'] = df[SIGNAL_COL].diff().fillna(0)
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df['CURRENT_FIX_INTEREST_AREA_LABEL'] = pd.to_numeric(df['CURRENT_FIX_INTEREST_AREA_LABEL'],
                                                          errors='coerce').fillna(-1).astype(int)

    train_df, test_df = split_train_test_for_time_series(df, test_size=0.2, random_state=seed,
                                                         input_columns=input_columns)
    train_df, val_df = split_train_test_for_time_series(train_df, test_size=0.2, random_state=seed,
                                                        input_columns=input_columns)

    gt_ailments_df = (
        test_df.loc[test_df['AILMENT_NUMBER'] != -1,
        ['RECORDING_SESSION_LABEL', 'TRIAL_INDEX', 'AILMENT_NUMBER']]
        .drop_duplicates()
    )
    total_unique_ailments_in_test = len(gt_ailments_df)

    # 2. Create windows
    X_train, Y_train, train_metadata, _ = create_dynamic_time_series_for_ekg_with_ailment(
        train_df, feature_columns, window_size=window_size, window_step=window_step
    )
    X_val, Y_val, val_metadata, _ = create_dynamic_time_series_for_ekg_with_ailment(
        val_df, feature_columns, window_size=window_size, window_step=window_step
    )
    X_test, Y_test, test_metadata, test_ailment_locations = create_dynamic_time_series_for_ekg_with_ailment(
        test_df, feature_columns, window_size=window_size, window_step=window_step
    )

    save_dir = f'results/ekg_ailment_tracking_participant_{participant_id}'
    os.makedirs(save_dir, exist_ok=True)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).permute(0, 2, 1)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).permute(0, 2, 1)

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    test_loader = DataLoader(TensorDataset(X_test_tensor, torch.tensor(Y_test, dtype=torch.long)), batch_size=32,
                             shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    ensemble_trainer = EnsembleTrainer(
        base_model_class=CNN1DModel,
        model_params={'input_dim': X_train_tensor.shape[1], 'window_size': window_size, 'output_classes': 2},
        n_models=10, device=device, save_path=os.path.join(save_dir, 'ensemble_models')
    )
    class_counts = np.bincount(Y_train)
    weights = 1.0 / class_counts

    print("Training ensemble for Stage 1...")
    ensemble_trainer.train_ensemble(
        train_dataset=train_dataset,
        val_loader=val_loader,
        batch_size=32,
        epochs=300,
        criterion=nn.CrossEntropyLoss(),
        optimizer_class=optim.Adam,
        optimizer_params={'lr': 0.001},
        majority_weight=weights[0] if len(weights) > 1 else 0.1
    )

    best_threshold = ensemble_trainer.find_best_threshold(val_loader)
    test_predictions = ensemble_trainer.predict(test_loader, threshold=best_threshold)
    step1_f1 = f1_score(Y_test, test_predictions, zero_division=0)
    step1_recall = recall_score(Y_test, test_predictions, zero_division=0)
    step1_precision = precision_score(Y_test, test_predictions, zero_division=0)


    cm = confusion_matrix(Y_test, test_predictions)
    print("Step 1 confusion matrix: {}\n".format(cm))
    print("Step 1 F1 score: {}\n".format(step1_f1))
    print("Step 1 Recall score: {}\n".format(step1_recall))
    print("Step 1 Precision score: {}\n".format(step1_precision))



    step1_positive_indices = np.where(test_predictions == 1)[0]

    # === MODIFIED: EKG cell-level density heatmaps loop ===
    # Iterate over ALL participants found in the test set to handle 'None' case
    try:
        # Identify all unique participants in the test set
        unique_participants = test_df['RECORDING_SESSION_LABEL'].unique()
        print(f"Generating density heatmaps for {len(unique_participants)} participants found in test set...")

        for pid in unique_participants:
            plot_ekg_spatial_stripes_panel(
                test_metadata=test_metadata,
                test_df=test_df,
                positive_indices=step1_positive_indices,
                out_dir=save_dir,
                participant_id=str(pid),
                ia_col="CURRENT_FIX_INTEREST_AREA_LABEL"
            )
    except Exception as e:
        print(f"[WARN] Failed to plot EKG density heatmaps: {e}")
        import traceback
        traceback.print_exc()

    # === Stage 2 Localization ===
    if len(step1_positive_indices) > 0:
        stage1_model_for_loc = ensemble_trainer.models[0]
        localizer = GradientLocalizer(stage1_model_for_loc, device)

        all_stage2_results = []
        for idx in step1_positive_indices:
            window_metadata = test_metadata[idx]
            window_tensor = X_test_tensor[idx].unsqueeze(0)
            predicted_row = localizer.localize_target(window_tensor.clone())
            true_target_positions = window_metadata.get('target_positions', [])
            is_correct = 1 if predicted_row in true_target_positions else 0

            ailment_found = 'None'
            for detail in window_metadata.get('ailment_details', []):
                if detail['relative_position'] == predicted_row:
                    ailment_found = detail['ailment_number']
                    break
            all_stage2_results.append({
                'window_index': idx,
                'predicted_row': predicted_row,
                'true_targets': true_target_positions,
                'is_correct': is_correct,
                'ailment_found': ailment_found,
                'ailments_in_window': window_metadata.get('ailment_numbers', [])
            })

        if all_stage2_results:
            final_df, _ = reconstruct_and_evaluate_efficiency(
                all_stage2_results, test_df, test_metadata, strategy='majority'
            )
            output_path = os.path.join(save_dir, 'final_predictions_MAJORTY.csv')
            final_df.to_csv(output_path, index=False)

            final_df_any, _ = reconstruct_and_evaluate_efficiency(
                all_stage2_results, test_df, test_metadata, strategy='any'
            )
            output_path = os.path.join(save_dir, 'final_predictions_ANY.csv')
            final_df_any.to_csv(output_path, index=False)

            # === Export unified per-row results ===
            out_rows_csv = Path(save_dir) / "row_level_results_majority_any.csv"
            export_row_prediction_table(
                test_df=test_df,
                final_df_majority=final_df,
                final_df_any=final_df_any,
                out_csv=out_rows_csv,
                id_cols=("RECORDING_SESSION_LABEL", "TRIAL_INDEX"),
                slice_col="CURRENT_FIX_COMPONENT_IMAGE_FILE",
                ailment_col="AILMENT_NUMBER",
                loc_col="CURRENT_FIX_INTEREST_AREA_LABEL",
            )

            tol_metrics = compute_tolerance_metrics(
                test_df=test_df,
                final_df=final_df,
                id_cols=("RECORDING_SESSION_LABEL", "TRIAL_INDEX"),
                ia_col="CURRENT_FIX_INTEREST_AREA_LABEL",
                tolerance=0
            )

            trial_metrics = compute_trial_level_tolerance_metrics(
                test_df=test_df,
                final_df=final_df,
                tolerance=0,  # +/- 1 Grid Location
                ia_col="CURRENT_FIX_INTEREST_AREA_LABEL"
            )

            n_flagged = trial_metrics["Trial_TP"] + trial_metrics["Trial_FP"]
            n_total = trial_metrics["Total_Trials"]

            print(f"\n[tolerance] Trial-Level Performance (Tolerance +/-1):")
            print(f"  Total ECGs: {n_total}")
            print(f"  Sent to Doctor: {n_flagged} ({trial_metrics['Review_Rate'] * 100:.1f}%)")
            print(f"    - Accurate Referrals (TP): {trial_metrics['Trial_TP']}")
            print(f"    - False Alarms (FP):       {trial_metrics['Trial_FP']}")
            print(f"  Filtered Out (Auto): {trial_metrics['Trial_TN'] + trial_metrics['Trial_FN']}")
            print(f"    - Correctly Filtered (TN): {trial_metrics['Trial_TN']}")
            print(f"    - Missed Patients (FN):    {trial_metrics['Trial_FN']}")
            print(f"  Trial F1 Score: {trial_metrics['Trial_F1']:.4f}")

            print(f"Tol:{tol_metrics}")



if __name__ == "__main__":
    window_size = 5
    seed = 42
    participant_id = None
    RUN_SUITE = False

    if RUN_SUITE:
        csv_path = Path(__file__).parent.parent / "EKG data" / "ML_ECG_Data.csv"

    else:
        try:
            ekg_two_step_pipeline(participant_id, window_size, seed)
            print("\nSUCCESS: EKG pipeline completed!")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            import traceback;

            traceback.print_exc()