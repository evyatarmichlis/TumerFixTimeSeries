from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu, f_oneway, kruskal
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns

cmap = colormaps['tab10']

# Example usage within your pipeline:
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def explore_data(selected_features, feature_to_test):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu, f_oneway, kruskal
    import pandas as pd
    import numpy as np
    import itertools

    # --- Data Preprocessing ---
    aggregated_features_norm = selected_features.copy()

    for col in feature_to_test:
        aggregated_features_norm[col] = pd.to_numeric(aggregated_features_norm[col], errors='coerce')
        aggregated_features_norm[col] = aggregated_features_norm[col].fillna(0)

    scaler = StandardScaler()
    aggregated_features_norm[feature_to_test] = scaler.fit_transform(aggregated_features_norm[feature_to_test])

    # Ensure the target column is numeric (only 0 and 1)
    aggregated_features_norm['target'] = aggregated_features_norm['target'].astype(int)

    # Create separate dataframes for each target group (only 0 and 1)
    df_target_0 = aggregated_features_norm[aggregated_features_norm['target'] == 0]
    df_target_1 = aggregated_features_norm[aggregated_features_norm['target'] == 1]

    aggregated_features_norm['TRIAL_INDEX'] = pd.to_numeric(aggregated_features_norm['TRIAL_INDEX'], errors='coerce')

    # Get the sorted unique trial indices
    # unique_trials = sorted(aggregated_features_norm['TRIAL_INDEX'].dropna().unique())
    #
    # for trial in unique_trials:
    #     df_trial = aggregated_features_norm[aggregated_features_norm['TRIAL_INDEX'] == trial]
    #     num_features = len(feature_to_test)
    #     fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(8, num_features * 3))
    #     if num_features == 1:
    #         axes = [axes]
    #
    #     for ax, feature in zip(axes, feature_to_test):
    #         # Plot distribution for normal class
    #         sns.histplot(df_trial[df_trial['target'] == 0][feature], color='blue', kde=True, stat="density",
    #                      label='normal', ax=ax, bins=10, alpha=0.6)
    #         # Plot distribution for non conscious class
    #         sns.histplot(df_trial[df_trial['target'] == 1][feature], color='red', kde=True, stat="density",
    #                      label='non conscious', ax=ax, bins=10, alpha=0.6)
    #         ax.set_title(f'Trial {trial} - Distribution of {feature}')
    #         ax.legend()
    #     plt.tight_layout()
    #     plt.show()

    # --- Extra Analysis: Pairwise Statistical Tests ---
    print("Pairwise Statistical Tests:")
    targets = {
        0: df_target_0,
        1: df_target_1,
    }

    def cohens_d(a, b):
        n1, n2 = len(a), len(b)
        s1, s2 = np.std(a, ddof=1), np.std(b, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
        return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else np.nan

    for feature in feature_to_test:
        print(f"\nResults for feature: {feature}")
        for target_a, target_b in itertools.combinations(targets.keys(), 2):
            data_a = targets[target_a][feature]
            data_b = targets[target_b][feature]
            t_stat, t_pval = ttest_ind(data_a, data_b, nan_policy='omit')
            ks_stat, ks_pval = ks_2samp(data_a, data_b)
            try:
                mw_stat, mw_pval = mannwhitneyu(data_a, data_b, alternative='two-sided')
            except Exception as e:
                mw_stat, mw_pval = np.nan, np.nan
            d = cohens_d(data_a, data_b)
            print(f"Targets {target_a} vs {target_b}:")
            print(f"  t-test: t = {t_stat:.3f}, p = {t_pval:.3e}")
            print(f"  KS test: KS = {ks_stat:.3f}, p = {ks_pval:.3e}")
            print(f"  Mann-Whitney U test: U = {mw_stat:.3f}, p = {mw_pval:.3e}")
            print(f"  Cohen's d: {d:.3f}")

    for feature in feature_to_test:
        groups = [df[feature].dropna().values for df in targets.values()]
        try:
            f_stat, f_pval = f_oneway(*groups)
        except Exception as e:
            f_stat, f_pval = np.nan, np.nan
        try:
            kw_stat, kw_pval = kruskal(*groups)
        except Exception as e:
            kw_stat, kw_pval = np.nan, np.nan
        print(f"\nGlobal tests for {feature}:")
        print(f"  ANOVA: F = {f_stat:.3f}, p = {f_pval:.3e}")
        print(f"  Kruskal-Wallis: H = {kw_stat:.3f}, p = {kw_pval:.3e}")

    # --- Additional Visualization: Box Plots by Target ---
    # Update the target mapping for 2 classes
    target_mapping = {'0': 'normal', '1': 'non conscious'}
    aggregated_features_norm['target_str'] = aggregated_features_norm['target'].astype(str)
    aggregated_features_norm['target_label'] = aggregated_features_norm['target_str'].map(target_mapping)

    for feature in feature_to_test:
        plt.figure(figsize=(8, 6))
        ax = sns.boxplot(x='target_label', y=feature, data=aggregated_features_norm,
                         palette={'normal': 'blue', 'non conscious': 'red'})
        plt.title(f'Box Plot of {feature} by Target')
        plt.xlabel('Target')
        plt.ylabel(feature)
        # Remove the legend if it exists
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        plt.show()


    # --- t-SNE Visualization ---
    # X = aggregated_features_norm[feature_to_test].values
    # tsne = TSNE(n_components=2, random_state=42)
    # tsne_result = tsne.fit_transform(X)
    # aggregated_features_norm['tsne-2d-one'] = tsne_result[:, 0]
    # aggregated_features_norm['tsne-2d-two'] = tsne_result[:, 1]
    #
    # colors = {0: 'blue', 1: 'red'}
    # plt.figure(figsize=(8, 8))
    # for target_value in aggregated_features_norm['target'].unique():
    #     indices = aggregated_features_norm['target'] == target_value
    #     plt.scatter(aggregated_features_norm.loc[indices, 'tsne-2d-one'],
    #                 aggregated_features_norm.loc[indices, 'tsne-2d-two'],
    #                 c=colors[target_value],
    #                 label=f'target {target_value}',
    #                 alpha=0.6)
    # plt.title("t-SNE Cluster Plot")
    # plt.xlabel("t-SNE Component 1")
    # plt.ylabel("t-SNE Component 2")
    # plt.legend()
    # plt.show()

    # --- Additional Analysis: Distribution by TRIAL_INDEX ---
    # Check if the distribution of each feature changes per TRIAL_INDEX.
    # Assumes that the TRIAL_INDEX column exists in aggregated_features_norm.
    # If TRIAL_INDEX is numeric, we sort them.
    aggregated_features_norm['TRIAL_INDEX'] = pd.to_numeric(aggregated_features_norm['TRIAL_INDEX'], errors='coerce')
    for feature in feature_to_test:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='TRIAL_INDEX', y=feature, data=aggregated_features_norm, palette="Set3")
        plt.title(f'Distribution of {feature} by TRIAL_INDEX')
        plt.xlabel('TRIAL_INDEX')
        plt.ylabel(feature)
        plt.show()

    # Ensure TRIAL_INDEX is numeric
    aggregated_features_norm['TRIAL_INDEX'] = pd.to_numeric(aggregated_features_norm['TRIAL_INDEX'], errors='coerce')

    # Identify the trials that contain at least one instance with target == 1.
    trials_with_target1 = aggregated_features_norm.groupby('TRIAL_INDEX')['target'].max().reset_index()
    trials_with_target1 = trials_with_target1[trials_with_target1['target'] == 1]['TRIAL_INDEX']

    # Filter the dataframe to include only these trials.
    filtered_df = aggregated_features_norm[aggregated_features_norm['TRIAL_INDEX'].isin(trials_with_target1)]

    # Define a custom palette mapping: target 0 to blue, target 1 to red
    custom_palette = {0: 'blue', 1: 'red'}

    # Plot boxplots for each feature, separating target 0 and target 1 via hue.
    for feature in feature_to_test:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='TRIAL_INDEX', y=feature, hue='target', data=filtered_df, palette=custom_palette)
        plt.title(f'Distribution of {feature} by TRIAL_INDEX (Only Trials with Target=1)')
        plt.xlabel('TRIAL_INDEX')
        plt.ylabel(feature)
        plt.legend(title='Target', labels=['normal (0)', 'non conscious (1)'])
        plt.show()

csv_path = Path(__file__).parent.parent / "fwd_data" / 'Nodule_Categorized_Fixation_Data_1_18.csv'
df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
df = df[df['RECORDING_SESSION_LABEL'] == 2]
df['target'] = np.where(df['LOCATION_TYPE'] == 'NODULE_HIT', 1, 0)
# Make sure TRIAL_INDEX is present as a column for the trial-based analysis
feature_to_test = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_COMPONENT_DURATION']
explore_data(df, feature_to_test)