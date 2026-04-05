import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

# --- Configuration ---
INPUT_CSV = 'results/ct_experiments/summary.csv' # <-- PATH UPDATED HERE
OUTPUT_DIR = 'plots_new_data'


def calculate_random_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the F1 score for a random classifier as a baseline.
    """
    rand_f1s = []
    for _, row in df.iterrows():
        if row['level'] in ['Window', 'Fixation Event'] and pd.notna(row['TP']):
            P = row['TP'] + row['FN']
            N = row['TP'] + row['FP'] + row['TN'] + row['FN']
            K = row['TP'] + row['FP']

            if N == 0 or (P == 0 and K == 0):
                rand_f1s.append(0)
                continue

            rand_precision = P / N if N > 0 else 0
            rand_recall = K / N if N > 0 else 0

            if rand_precision + rand_recall > 0:
                rand_f1 = 2 * (rand_precision * rand_recall) / (rand_precision + rand_recall)
            else:
                rand_f1 = 0
            rand_f1s.append(rand_f1)
        else:
            rand_f1s.append(np.nan)

    df['random_f1'] = rand_f1s
    return df


def plot_within_participant(df: pd.DataFrame, output_dir: str):
    """
    Generates bar plots for within-participant experiments.
    """
    print("Plotting 'within_participant' scope...")
    sub_df = df[df['scope'] == 'within_participant'].copy()

    valid_participants = sub_df.groupby('experiment')['f1'].sum() > 0
    sub_df = sub_df[sub_df['experiment'].isin(valid_participants[valid_participants].index)]

    if sub_df.empty:
        print("  - No valid data to plot for this scope.")
        return

    try:
        sub_df['exp_num'] = sub_df['experiment'].str.extract(r'(\d+)').astype(int)
        order = sub_df.sort_values('exp_num')['experiment'].unique()
    except:
        order = sorted(sub_df['experiment'].unique())

    g = sns.catplot(
        data=sub_df,
        x='experiment',
        y='f1',
        hue='level',
        col='strategy',
        kind='bar',
        height=6,
        aspect=1.8,
        order=order
    )

    bar_handles = g.legend.legend_handles
    bar_labels = [t.get_text() for t in g.legend.texts]
    g.legend.remove()

    g.fig.suptitle('Within-Participant Performance (Train & Test on same Participant)', y=1.03, fontsize=16)
    g.set_axis_labels('Participant ID', 'F1 Score')
    g.set_titles('Strategy: {col_name}')
    g.set_xticklabels(rotation=45, ha='right')

    for ax in g.axes.flat:
        strat = ax.get_title().split(': ')[1]
        mean_random_f1 = sub_df[(sub_df['strategy'] == strat) & (sub_df['level'] != 'Ailment')]['random_f1'].mean()
        if pd.notna(mean_random_f1):
            ax.axhline(mean_random_f1, ls='--', color='red', lw=1.5, label=f'Avg. Random F1 ({mean_random_f1:.2f})')

    line_handle, line_label = g.axes.flat[0].get_legend_handles_labels()

    all_handles = bar_handles + line_handle
    all_labels = bar_labels + line_label
    g.fig.legend(all_handles, all_labels, loc='upper right', bbox_to_anchor=(0.98, 0.95), title="Metric Level")

    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    plt.savefig(os.path.join(output_dir, '1_within_participant_f1_score.png'), dpi=300)
    plt.close()


def plot_cross_participant(df: pd.DataFrame, output_dir: str):
    """
    Generates heatmaps for cross-participant generalization.
    """
    print("Plotting 'cross_participant' scope...")
    sub_df = df[df['scope'] == 'cross_participant'].copy()

    sub_df[['train_p', 'test_p']] = sub_df['experiment'].str.extract(r'train_(\d+).*test_(\d+)')

    if sub_df['train_p'].isnull().all():
        print("  - Could not parse train/test participants. Skipping.")
        return

    train_ids = sub_df['train_p'].dropna().unique()
    test_ids = sub_df['test_p'].dropna().unique()
    all_ids_int = sorted([int(i) for i in set(train_ids) | set(test_ids)])
    participants = [str(i) for i in all_ids_int]

    for strategy in sub_df['strategy'].unique():
        for level in sub_df['level'].unique():

            plot_df = sub_df[(sub_df['strategy'] == strategy) & (sub_df['level'] == level)]
            if plot_df.empty: continue

            pivot_df = plot_df.pivot_table(
                index='train_p',
                columns='test_p',
                values='f1',
                aggfunc='mean'
            ).reindex(index=participants, columns=participants)

            plt.figure(figsize=(14, 12))
            sns.heatmap(
                pivot_df,
                annot=True,
                fmt=".3f",
                cmap="viridis",
                linewidths=.5,
                cbar_kws={'label': 'F1 Score'}
            )

            title = f'Cross-Participant Generalization F1 Score\nStrategy: {strategy} | Level: {level}'
            plt.title(title, fontsize=16)
            plt.xlabel('Test Participant', fontsize=12)
            plt.ylabel('Train Participant', fontsize=12)
            plt.tight_layout()

            filename = f'2_cross_participant_{strategy}_{level}.png'
            plt.savefig(os.path.join(output_dir, filename), dpi=300)
            plt.close()


def plot_other_scopes(df: pd.DataFrame, scope_name: str, output_dir: str, plot_num: int):
    """
    Generic plotting function for scopes with a small number of experiments.
    """
    print(f"Plotting '{scope_name}' scope...")
    sub_df = df[df['scope'] == scope_name].copy()

    if sub_df.empty:
        print(f"  - No data to plot for scope '{scope_name}'.")
        return

    g = sns.catplot(
        data=sub_df,
        x='experiment',
        y='f1',
        hue='level',
        col='strategy',
        kind='bar',
        height=6,
        aspect=1.2
    )

    bar_handles = g.legend.legend_handles
    bar_labels = [t.get_text() for t in g.legend.texts]
    g.legend.remove()

    title = scope_name.replace('_', ' ').title()
    g.fig.suptitle(f'{title} Performance', y=1.03, fontsize=16)
    g.set_axis_labels('Experiment', 'F1 Score')
    g.set_titles('Strategy: {col_name}')
    g.set_xticklabels(rotation=15, ha='right')

    for ax in g.axes.flat:
        strat = ax.get_title().split(': ')[1]
        mean_random_f1 = sub_df[(sub_df['strategy'] == strat) & (sub_df['level'] != 'Ailment')]['random_f1'].mean()
        if pd.notna(mean_random_f1):
            ax.axhline(mean_random_f1, ls='--', color='red', lw=1.5, label=f'Avg. Random F1 ({mean_random_f1:.2f})')

    line_handle, line_label = g.axes.flat[0].get_legend_handles_labels()

    all_handles = bar_handles + line_handle
    all_labels = bar_labels + line_label
    g.fig.legend(all_handles, all_labels, loc='upper right', bbox_to_anchor=(0.95, 0.92), title="Metric Level")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    filename = f'{plot_num}_{scope_name}_f1_score.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()


def main():
    """
    Main function to load data, process it, and generate all plots.
    """
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file not found at '{INPUT_CSV}'")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sns.set_theme(style="whitegrid")

    df = pd.read_csv(INPUT_CSV)
    df['level'] = df['level'].replace({'row': 'Fixation Event'}).str.title()
    df = calculate_random_baseline(df)

    plot_within_participant(df, OUTPUT_DIR)
    plot_cross_participant(df, OUTPUT_DIR)
    plot_other_scopes(df, 'group_of_k_vs_rest', OUTPUT_DIR, plot_num=3)
    plot_other_scopes(df, 'train_all', OUTPUT_DIR, plot_num=4)

    print(f"\nAll plots have been generated and saved in the '{OUTPUT_DIR}' directory.")


if __name__ == '__main__':
    main()