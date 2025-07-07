from collections import Counter

from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu, f_oneway, kruskal
import pandas as pd
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from sklearn.ensemble import RandomForestClassifier
import statsmodels.api as sm
from sklearn.manifold import TSNE
from statsmodels.formula.api import ols

from representation_method.utils.data_loader import load_eye_tracking_data, DataConfig

cmap = colormaps['tab10']

import seaborn as sns

# Example usage within your pipeline:
from pathlib import Path


from sklearn.decomposition import PCA
import umap.umap_ as umap  # make sure to install umap-learn
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

def explore_data(selected_features, feature_to_test):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import ttest_ind, f_oneway, kruskal
    import pandas as pd
    import numpy as np

    # # --- Data Preprocessing ---
    aggregated_features_norm = selected_features.copy()

    for col in feature_to_test:
        aggregated_features_norm[col] = pd.to_numeric(aggregated_features_norm[col], errors='coerce')
        aggregated_features_norm[col] = aggregated_features_norm[col].fillna(0)

    scaler = StandardScaler()
    aggregated_features_norm[feature_to_test] = scaler.fit_transform(aggregated_features_norm[feature_to_test])

    # Ensure the target column is numeric
    aggregated_features_norm['target'] = aggregated_features_norm['target'].astype(int)

    # Create separate dataframes for each target group
    df_target_0 = aggregated_features_norm[aggregated_features_norm['target'] == 0]
    df_target_1 = aggregated_features_norm[aggregated_features_norm['target'] == 1]

    num_features = len(feature_to_test)
    fig, axes = plt.subplots(nrows=num_features, ncols=1, figsize=(8, num_features * 3))
    if num_features == 1:
        axes = [axes]

    for ax, feature in zip(axes, feature_to_test):
        sns.histplot(df_target_0[feature], color='blue', kde=True, stat="density",   palette={0: 'blue', 1: 'red'},

                     label='normal', ax=ax, bins=10, alpha=0.6)
        sns.histplot(df_target_1[feature], color='red', kde=True, stat="density",        palette={0: 'blue', 1: 'red'},

                     label='non conscious', ax=ax, bins=10, alpha=0.6)
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
    plt.tight_layout()
    # plt.show()

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

    print("\n--- Trail-level Statistical Tests ---")
    for trail in aggregated_features_norm['TRIAL_INDEX'].unique():
        trail_data = aggregated_features_norm[aggregated_features_norm['TRIAL_INDEX'] == trail]

        # Skip trails with no target=1 or target=0 samples
        if 1 not in trail_data['target'].values or 0 not in trail_data['target'].values:
            print(f"Trail {trail}: Skipping (missing one of the classes)")
            continue

        print(f"\nTrail {trail}:")
        trail_0 = trail_data[trail_data['target'] == 0]
        trail_1 = trail_data[trail_data['target'] == 1]

        for feature in feature_to_test:
            t_stat, p_val = ttest_ind(
                trail_0[feature].dropna(),
                trail_1[feature].dropna(),
                equal_var=False
            )
            effect_size = cohens_d(trail_0[feature].dropna(), trail_1[feature].dropna())
            print(f"  {feature}: t={t_stat:.3f}, p={p_val:.3e}, Cohen's d={effect_size:.3f}")
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
    # Feature distribution comparison across trails
    # before your “for feature in feature_to_test:” loop
    for feature in feature_to_test:
        # 1) Find trails that have both classes
        trails_with_both = []
        for t in sorted(aggregated_features_norm['TRIAL_INDEX'].unique()):
            td = aggregated_features_norm[aggregated_features_norm['TRIAL_INDEX'] == t]
            if set(td['target']) == {0, 1}:
                trails_with_both.append(t)
        if not trails_with_both:
            print(f"No trails with both classes for {feature}")
            continue

        # 2) Split into two halves
        half = len(trails_with_both) // 2
        first_trails = trails_with_both[:half]
        second_trails = trails_with_both[half:]

        # 3) Plot first half
        df1 = aggregated_features_norm[aggregated_features_norm['TRIAL_INDEX'].isin(first_trails)]
        plt.figure(figsize=(12, 6))
        sns.violinplot(
            x='TRIAL_INDEX', y=feature, hue='target',
            data=df1, palette={0: 'blue', 1: 'red'},
            split=True, inner='quart'
        )
        plt.title(f'{feature} — Trails {first_trails[0]} to {first_trails[-1]}')
        plt.legend(title='Target', loc='upper right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # 4) Plot second half
        df2 = aggregated_features_norm[aggregated_features_norm['TRIAL_INDEX'].isin(second_trails)]
        plt.figure(figsize=(12, 6))
        sns.violinplot(
            x='TRIAL_INDEX', y=feature, hue='target',
            data=df2, palette={0: 'blue', 1: 'red'},
            split=True, inner='quart'
        )
        plt.title(f'{feature} — Trails {second_trails[0]} to {second_trails[-1]}')
        plt.legend(title='Target', loc='upper right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Add simple t-test for global comparison between classes
    print("\n--- Global T-test Analysis ---")
    for feature in feature_to_test:
        # Get data for each class
        class0_data = aggregated_features_norm[aggregated_features_norm['target'] == 0][feature].dropna()
        class1_data = aggregated_features_norm[aggregated_features_norm['target'] == 1][feature].dropna()

        # Perform t-test
        t_stat, p_val = ttest_ind(class0_data, class1_data, equal_var=False)  # Using Welch's t-test

        # Calculate effect size (Cohen's d)
        effect_size = cohens_d(class0_data, class1_data)

        # Print results
        print(f"\nFeature: {feature}")
        print(f"  T-statistic: {t_stat:.3f}")
        print(f"  P-value: {p_val:.6f}")
        print(f"  Cohen's d: {effect_size:.3f}")
        print(f"  Class 0 mean: {class0_data.mean():.3f}, std: {class0_data.std():.3f}, n={len(class0_data)}")
        print(f"  Class 1 mean: {class1_data.mean():.3f}, std: {class1_data.std():.3f}, n={len(class1_data)}")

        # Interpretation
        if p_val < 0.05:
            print(f"  Result: Significant difference (p={p_val:.6f})")
            if effect_size > 0.8:
                print("  Effect size: Large")
            elif effect_size > 0.5:
                print("  Effect size: Medium")
            elif effect_size > 0.2:
                print("  Effect size: Small")
            else:
                print("  Effect size: Negligible")
        else:
            print(f"  Result: No significant difference (p={p_val:.6f})")

    print("\n--- Variance Component Analysis ---")
    for feature in feature_to_test:
        formula = f"{feature} ~ C(target) + C(TRIAL_INDEX) + C(target):C(TRIAL_INDEX)"
        model = ols(formula, data=aggregated_features_norm).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)

        # Calculate sum of squares proportions
        ss_total = aov_table['sum_sq'].sum()
        ss_target = aov_table['sum_sq'][0]
        ss_trail = aov_table['sum_sq'][1]
        ss_interaction = aov_table['sum_sq'][2]
        ss_residual = aov_table['sum_sq'][3] if len(aov_table['sum_sq']) > 3 else 0

        print(f"\nFeature: {feature}")
        print(f"Variance explained by target class: {ss_target / ss_total:.2%}")
        print(f"Variance explained by trail: {ss_trail / ss_total:.2%}")
        print(f"Variance explained by target-trail interaction: {ss_interaction / ss_total:.2%}")
        print(f"Unexplained variance: {ss_residual / ss_total:.2%}")
        print("\nANOVA table:")
        print(aov_table)
    # --- Additional Visualization: Box Plots by Target ---
    # Create a string version of target for the box plot
    target_mapping = {'0': 'normal', '1': 'non conscious', '2': 'all conscious', '3': 'fwd conscious'}

    # Convert target to string if not already, then create a new column with the labels
    aggregated_features_norm['target_str'] = aggregated_features_norm['target'].astype(str)
    aggregated_features_norm['target_label'] = aggregated_features_norm['target_str'].map(target_mapping)


    X = aggregated_features_norm[feature_to_test].values
    y_class = aggregated_features_norm['target'].values
    y_trail = aggregated_features_norm['TRIAL_INDEX'].values

    # Create color maps
    # For class (just 2 colors)
    class_colors = {0: 'blue', 1: 'red'}
    class_color_map = [class_colors[c] for c in y_class]

    # For trails (need many colors)
    unique_trails = np.unique(y_trail)
    trail_cmap = plt.cm.get_cmap('tab20', len(unique_trails))
    trail_color_dict = {trail: trail_cmap(i) for i, trail in enumerate(unique_trails)}
    trail_color_map = [trail_color_dict[t] for t in y_trail]

    # Filter data to include only trails that have both classes
    def filter_data_for_both_classes(X, y_class, y_trail):
        trails_with_both_classes = []

        for trail in np.unique(y_trail):
            trail_mask = y_trail == trail
            trail_classes = y_class[trail_mask]

            # Check if this trail has both classes
            if len(np.unique(trail_classes)) == 2:
                trails_with_both_classes.append(trail)

        if not trails_with_both_classes:
            print("Warning: No trails have both classes. Using all data.")
            return X, y_class, y_trail, np.unique(y_trail)

        # Create mask for selected trails
        selected_mask = np.isin(y_trail, trails_with_both_classes)

        print(f"Selected {len(trails_with_both_classes)} trails with both classes: {sorted(trails_with_both_classes)}")

        return X[selected_mask], y_class[selected_mask], y_trail[selected_mask], trails_with_both_classes

    # Filter the data
    X_filtered, y_class_filtered, y_trail_filtered, selected_trails = filter_data_for_both_classes(X, y_class, y_trail)

    # Update color maps for filtered data
    class_colors = {0: 'blue', 1: 'red'}
    class_color_map = [class_colors[c] for c in y_class_filtered]

    # Create color map for trails
    unique_trails = np.unique(y_trail_filtered)
    trail_cmap = plt.cm.get_cmap('tab20', len(unique_trails))
    trail_color_dict = {trail: trail_cmap(i) for i, trail in enumerate(unique_trails)}
    trail_color_map = [trail_color_dict[t] for t in y_trail_filtered]

    def plot_dimension_reduction(X_reduced, title, technique_name, y_class, y_trail, top_n=3):
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        # 1. Build DataFrame for embedding and labels
        df = pd.DataFrame(X_reduced, columns=['Dim1', 'Dim2'])
        df['class'] = y_class
        df['trail'] = y_trail

        # 2. Identify top N trails by sample count
        top_trails = df['trail'].value_counts().nlargest(top_n).index.tolist()
        df['trail_group'] = df['trail'].apply(lambda t: t if t in top_trails else 'Other')

        # 3. Prepare color maps
        # Class colors (0: blue, 1: red)
        class_colors = {0: 'blue', 1: 'red'}
        df['class_color'] = df['class'].map(class_colors)

        # Trail-group colors using a qualitative palette, with RGBA for transparency
        groups = df['trail_group'].unique().tolist()
        cmap = plt.cm.get_cmap('tab10', len(groups))
        trail_group_colors = {}
        for i, grp in enumerate(groups):
            base = cmap(i)
            # RGBA: last value is alpha
            trail_group_colors[grp] = (base[0], base[1], base[2], 1.0)

        # 4. Plot side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot 1: Colored by class
        ax1.scatter(df['Dim1'], df['Dim2'], c=df['class_color'], alpha=0.7, s=50)
        ax1.set_title(f'{title} - Colored by Class')
        ax1.set_xlabel(f'{technique_name} Component 1')
        ax1.set_ylabel(f'{technique_name} Component 2')
        class_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=col,
                                markersize=10, label=f'Class {c}')
                         for c, col in class_colors.items()]
        ax1.legend(handles=class_handles, title='Class')

        # Plot 2: Colored by trail-group (top N + Other) with transparency for 'Other'
        for grp in groups:
            mask = df['trail_group'] == grp
            alpha_val = 0.1 if grp == 'Other' else 1.0
            color = trail_group_colors[grp][:3]  # RGB for scatter
            ax2.scatter(
                df.loc[mask, 'Dim1'], df.loc[mask, 'Dim2'],
                color=[color], alpha=alpha_val, s=50
            )
        ax2.set_title(f'{title} - Colored by Trail Groups (Top {top_n} + Other)')
        ax2.set_xlabel(f'{technique_name} Component 1')
        ax2.set_ylabel(f'{technique_name} Component 2')
        trail_handles = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=trail_group_colors[grp],
                   markersize=10, label=str(grp))
            for grp in groups
        ]
        ax2.legend(handles=trail_handles, title='Trail Group', bbox_to_anchor=(1.05, 1))

        plt.tight_layout()
        plt.savefig(f'{technique_name}_top{top_n}_trails_visualization.png', dpi=300)
        plt.show()

    # Now apply dimensionality reduction techniques to the filtered data


    print("Computing PCA on filtered data...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_filtered)
    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    plot_dimension_reduction(X_pca, 'PCA Visualization (Trails with Both Classes)', 'PCA',y_class_filtered, y_trail_filtered)

    # 2. t-SNE
    print("Computing t-SNE on filtered data (this may take a moment)...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=20,  # try values between ~5 and 50
        learning_rate=200,  # try from 50 up to 1e3 (often ~200–500)
        early_exaggeration=12.0,  # default 12.0; higher → more initial cluster separation
        n_iter=1000,  # increase to 2000+ for better convergence
        metric='euclidean'  # you can also try 'cosine', 'manhattan', etc.
    )
    X_tsne = tsne.fit_transform(X_filtered)
    plot_dimension_reduction(X_tsne, 't-SNE Visualization (Trails with Both Classes)', 't-SNE',y_class_filtered, y_trail_filtered)


    # Bonus: Create a metric to quantify the clustering by class vs. by trail
    from sklearn.metrics import silhouette_score

    try:
        print("\nQuantifying clustering quality on filtered data:")

        # Only calculate if we have enough samples
        if len(X_filtered) > 10:
            class_silhouette = silhouette_score(X_pca, y_class_filtered)
            trail_silhouette = silhouette_score(X_pca, y_trail_filtered)

            print(f"PCA - Silhouette score by class: {class_silhouette:.3f}")
            print(f"PCA - Silhouette score by trail: {trail_silhouette:.3f}")
            print(f"Clustering is stronger by: {'TRAIL' if trail_silhouette > class_silhouette else 'CLASS'}")

            class_silhouette = silhouette_score(X_tsne, y_class_filtered)
            trail_silhouette = silhouette_score(X_tsne, y_trail_filtered)

            print(f"t-SNE - Silhouette score by class: {class_silhouette:.3f}")
            print(f"t-SNE - Silhouette score by trail: {trail_silhouette:.3f}")
            print(f"Clustering is stronger by: {'TRAIL' if trail_silhouette > class_silhouette else 'CLASS'}")


    except Exception as e:
        print(f"Couldn't compute silhouette scores: {e}")


csv_path = Path(__file__).parent.parent / "EKG data" / "ML_ECG_Data.csv"
df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
df['target'] = np.where(df['LOCATION_TYPE'] == 'MI_HIT', 1, 0)
df = df[df['RECORDING_SESSION_LABEL'] == 'P001']

print(df.columns)
quit(1)
# Load the participant's data
# config = DataConfig(
#     data_path='data/Categorized_Fixation_Data_1_18.csv',
#     approach_num=6,
#     normalize=True,
#     per_slice_target=True,
#     participant_id=1
# )
#
# df = load_eye_tracking_data(
#     data_path=config.data_path,
#     approach_num=config.approach_num,
#     participant_id=config.participant_id,
#     data_format="legacy"
# )


# Filter rows: keep only those where CURRENT_FIX_INTEREST_AREA_LABEL is a number between 1 and 15
# df = df[df['CURRENT_FIX_INTEREST_AREA_LABEL'].apply(lambda x: str(x).isdigit())]
# df['CURRENT_FIX_INTEREST_AREA_LABEL'] = df['CURRENT_FIX_INTEREST_AREA_LABEL'].astype(int)
# df = df[df['CURRENT_FIX_INTEREST_AREA_LABEL'].between(1, 15)]
#
# # Function that counts visits per area in each trial:
# # It counts how many times each area is visited in the trial.
# def add_visit_counts(group):
#     counts = group['CURRENT_FIX_INTEREST_AREA_LABEL'].value_counts().to_dict()
#     max_visits = max(counts.values()) if counts else 0
#     group['VISIT_COUNT'] = group['CURRENT_FIX_INTEREST_AREA_LABEL'].map(counts)
#     group['MAX_VISIT'] = max_visits
#     return group
#
# # Apply the function groupwise and reset the index without dropping the grouping columns
# df = df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'], group_keys=False).apply(add_visit_counts).reset_index()
#
# # Create a wide-format DataFrame that has one column per area (AREA_1, AREA_2, ... AREA_15)
# wide_counts = (
#     df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX', 'CURRENT_FIX_INTEREST_AREA_LABEL'])
#       .size()
#       .unstack(fill_value=0)
# )
# wide_counts = wide_counts.rename(columns=lambda x: f"AREA_{x}")
#
# # Merge the wide-format counts back into the original DataFrame
# df = df.merge(wide_counts, on=['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'], how='left')
base_features = ['Pupil_Size', 'CURRENT_FIX_DURATION']
# wide_features = [f"AREA_{i}" for i in range(1, 16)]
feature_to_test = base_features
explore_data(df, feature_to_test)

quit()
#
# #
# #
# #
# #
# #
# #
#
# df_trail = df
# # explore_data(df_trail, feature_to_test)
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import LeaveOneOut
# from sklearn.metrics import classification_report
#
# # For demonstration, we assume df_trail contains your subdataset and features:
# # 'VISIT_COUNT' and 'CURRENT_FIX_DURATION'
# base_features = ['Pupil_Size', 'CURRENT_FIX_DURATION']
# features = base_features
#
# # Prepare X (features) and y (target), filling any missing values with 0
# X = df_trail[features].fillna(0)
# y = df_trail['target']
#
# loo = LeaveOneOut()
# predictions = []
# actual = []
#
# for train_index, test_index in loo.split(X):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#     # Create and train the model
#     model = LogisticRegression(random_state=42)
#     model.fit(X_train, y_train)
#
#     # Predict on the held-out sample
#     y_pred = model.predict(X_test)
#     predictions.append(y_pred[0])
#     actual.append(y_test.values[0])
#
# print("Logistic Regression LOOCV Classification Report:")
# print(classification_report(actual, predictions))
# from sklearn.model_selection import LeaveOneGroupOut
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
#
# # Define your features
# base_features = ['Pupil_Size', 'CURRENT_FIX_DURATION']
# # wide_features = [f"AREA_{i}" for i in range(1, 16)]
# features = base_features
#
# # Prepare X and y
# X = df_trail[features].fillna(0)
# y = df_trail['target']
#
# # Define groups: one per (RECORDING_SESSION_LABEL, TRIAL_INDEX_)
# groups = df_trail['RECORDING_SESSION_LABEL'].astype(str) + "_" + df_trail['TRIAL_INDEX'].astype(str)
#
# # Initialize Leave-One-Group-Out CV
# logo = LeaveOneGroupOut()
#
# predictions = []
# actual = []
#
# for train_index, test_index in logo.split(X, y, groups=groups):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#     # Train model
#     model = LogisticRegression(random_state=42, max_iter=1000)
#     model.fit(X_train, y_train)
#
#     # Predict
#     y_pred = model.predict(X_test)
#     predictions.extend(y_pred)
#     actual.extend(y_test)
#
# # Classification report
# print("Logistic Regression Grouped LOOCV Classification Report:")
# print(classification_report(actual, predictions))
#
# quit()
# from sklearn.svm import SVC
# from sklearn.metrics import classification_report
# from sklearn.model_selection import LeaveOneOut
#
# predictions = []
# actual = []
#
# for train_index, test_index in loo.split(X):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#     # Create and train an SVM model with a linear kernel
#     svm_model = SVC(kernel='rbf', probability=True, random_state=42)
#     svm_model.fit(X_train, y_train)
#
#     # Predict on the held-out sample
#     y_pred = svm_model.predict(X_test)
#     predictions.append(y_pred[0])
#     actual.append(y_test.values[0])
#
# print("SVM LOOCV Classification Report:")
# print(classification_report(actual, predictions))
#
#
#
# loo = LeaveOneOut()
# predictions = []
# actual = []
#
# for train_index, test_index in loo.split(X):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#     # Create and train the RandomForest model
#     model = RandomForestClassifier(random_state=42)
#     model.fit(X_train, y_train)
#
#     # Predict on the held-out sample
#     y_pred = model.predict(X_test)
#     predictions.append(y_pred[0])
#     actual.append(y_test.values[0])
#
# print("RF LOOCV Classification Report:")
# print(classification_report(actual, predictions))
#
#
#
# # Define feature columns: basic features + wide-format area counts
# base_features = ['Pupil_Size', 'CURRENT_FIX_DURATION']
# # wide_features = [f"AREA_{i}" for i in range(1, 16)]
# features = base_features
#
# # Prepare X (features) and y (target), filling any missing values with 0
# X = df[features].fillna(0)
# y = df['target']
#
#
# split_with_groups = False
#
#
# if split_with_groups:
#     # Use original df to get grouping info
#     groups = df['RECORDING_SESSION_LABEL'].astype(str) + "_" + df['TRIAL_INDEX'].astype(str)
#
#     gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
#     train_idx, test_idx = next(gss.split(X, y, groups=groups))
#
#     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
#     # Extract the trials from the original df based on indices
#     train_trials = df.iloc[train_idx][['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']].drop_duplicates()
#     test_trials = df.iloc[test_idx][['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']].drop_duplicates()
#     # Get sets of (RECORDING_SESSION_LABEL, TRIAL_INDEX_) pairs
#     train_trial_set = set(map(tuple, train_trials.values))
#     test_trial_set = set(map(tuple, test_trials.values))
#
#     # Find intersection
#     intersection = train_trial_set.intersection(test_trial_set)
#
#     # Check and print result
#     if intersection:
#         print("⚠️ There IS an intersection between train and test trials:")
#         print(intersection)
#     else:
#         print("✅ No intersection between train and test trials. Data split is clean.")
# else:
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#
#
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
#
# # Initialize and train the LightGBM classifier on the oversampled data
# clf = lgb.LGBMClassifier(random_state=42)
# clf.fit(X_train_smote, y_train_smote)
#
# # Predict on the test set (which remains imbalanced)
# y_pred = clf.predict(X_test)
#
# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred)
#
# print("Accuracy:", accuracy)
# print("Classification Report:")
# print(report)

# import lightgbm as lgb
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.metrics import classification_report
# import numpy as np
#
# # Define your features
# features = ['Pupil_Size', 'CURRENT_FIX_DURATION']
#
# # Prepare X (features) and y (target); fill missing values with 0
# X = df[features].fillna(0)
# y = df['target']
#
# # Create a pipeline that applies SMOTE then trains a LightGBM classifier.
# pipeline = Pipeline([
#     ('smote', SMOTE(random_state=42)),
#     ('clf', lgb.LGBMClassifier(random_state=42))
# ])
#
# # Define a stratified k-fold cross validation
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
#
# # Compute cross-validation accuracy scores
# accuracy_scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=skf)
# print("Cross-Validation Accuracy Scores:", accuracy_scores)
# print("Mean Accuracy:", np.mean(accuracy_scores))

# Optionally, display detailed classification reports for each fold:
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define your features
base_features = ['Pupil_Size', 'CURRENT_FIX_DURATION']
features = base_features

# Prepare X and y
X = df[features].fillna(0)
y = df['target']

# Define group labels (participant + trial)
groups = df['RECORDING_SESSION_LABEL'].astype(str) + "_" + df['TRIAL_INDEX'].astype(str)

# Initialize metrics storage
acc_scores = []
precision_scores_class0 = []
recall_scores_class0 = []
f1_scores_class0 = []
precision_scores_class1 = []
recall_scores_class1 = []
f1_scores_class1 = []

# Stratified K-fold with grouping
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

for train_index, test_index in sgkf.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Apply SMOTE
    X_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)

    # Handle imbalance
    counter = Counter(y_train)
    scale_pos_weight = counter[0] / counter[1]

    # Train LightGBM
    clf = lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
    clf.fit(X_train_smote, y_train_smote)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc_scores.append(acc)

    prec = precision_score(y_test, y_pred, average=None, labels=[0, 1])
    rec = recall_score(y_test, y_pred, average=None, labels=[0, 1])
    f1 = f1_score(y_test, y_pred, average=None, labels=[0, 1])

    precision_scores_class0.append(prec[0])
    recall_scores_class0.append(rec[0])
    f1_scores_class0.append(f1[0])
    precision_scores_class1.append(prec[1])
    recall_scores_class1.append(rec[1])
    f1_scores_class1.append(f1[1])

    print(f"Fold {fold} - Accuracy: {acc*100:.2f}%")
    print(f"    Class 0 - Precision: {prec[0]*100:.2f}%, Recall: {rec[0]*100:.2f}%, F1: {f1[0]*100:.2f}%")
    print(f"    Class 1 - Precision: {prec[1]*100:.2f}%, Recall: {rec[1]*100:.2f}%, F1: {f1[1]*100:.2f}%")
    fold += 1

# Overall performance summary
acc_mean = np.mean(acc_scores)
acc_std = np.std(acc_scores)

print(f"\nOverall Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")

# Compute averages and standard deviations for class 0 metrics
prec0_mean = np.mean(precision_scores_class0)
prec0_std = np.std(precision_scores_class0)
rec0_mean = np.mean(recall_scores_class0)
rec0_std = np.std(recall_scores_class0)
f1_0_mean = np.mean(f1_scores_class0)
f1_0_std = np.std(f1_scores_class0)

# Compute averages and standard deviations for class 1 metrics
prec1_mean = np.mean(precision_scores_class1)
prec1_std = np.std(precision_scores_class1)
rec1_mean = np.mean(recall_scores_class1)
rec1_std = np.std(recall_scores_class1)
f1_1_mean = np.mean(f1_scores_class1)
f1_1_std = np.std(f1_scores_class1)

print("\nAverage Results:")
print(f"Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")

print("\nClass 0 Metrics:")
print(f"  Precision: {prec0_mean*100:.2f}% ± {prec0_std*100:.2f}%")
print(f"  Recall:    {rec0_mean*100:.2f}% ± {rec0_std*100:.2f}%")
print(f"  F1-score:  {f1_0_mean*100:.2f}% ± {f1_0_std*100:.2f}%")

print("\nClass 1 Metrics:")
print(f"  Precision: {prec1_mean*100:.2f}% ± {prec1_std*100:.2f}%")
print(f"  Recall:    {rec1_mean*100:.2f}% ± {rec1_std*100:.2f}%")
print(f"  F1-score:  {f1_1_mean*100:.2f}% ± {f1_1_std*100:.2f}%")


# Define your features
base_features = ['Pupil_Size', 'CURRENT_FIX_DURATION']
features = base_features

# Prepare X and y
X = df[features].fillna(0)
y = df['target']

# Initialize metrics storage
acc_scores = []
precision_scores_class0 = []
recall_scores_class0 = []
f1_scores_class0 = []
precision_scores_class1 = []
recall_scores_class1 = []
f1_scores_class1 = []

# Stratified KFold random split
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold = 1

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Apply SMOTE to balance the training set
    X_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)

    # Compute scale_pos_weight based on the original training distribution (optional)
    counter = Counter(y_train)
    scale_pos_weight = counter[0] / counter[1]

    # Train LightGBM classifier
    clf = lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
    clf.fit(X_train_smote, y_train_smote)

    # Predict and evaluate metrics on test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acc_scores.append(acc)

    prec = precision_score(y_test, y_pred, average=None, labels=[0, 1])
    rec = recall_score(y_test, y_pred, average=None, labels=[0, 1])
    f1 = f1_score(y_test, y_pred, average=None, labels=[0, 1])

    precision_scores_class0.append(prec[0])
    recall_scores_class0.append(rec[0])
    f1_scores_class0.append(f1[0])
    precision_scores_class1.append(prec[1])
    recall_scores_class1.append(rec[1])
    f1_scores_class1.append(f1[1])

    print(f"Fold {fold} - Accuracy: {acc*100:.2f}%")
    print(f"    Class 0 - Precision: {prec[0]*100:.2f}%, Recall: {rec[0]*100:.2f}%, F1: {f1[0]*100:.2f}%")
    print(f"    Class 1 - Precision: {prec[1]*100:.2f}%, Recall: {rec[1]*100:.2f}%, F1: {f1[1]*100:.2f}%")
    fold += 1

# Compute overall performance metrics
acc_mean = np.mean(acc_scores)
acc_std = np.std(acc_scores)

print(f"\nOverall Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")

# Compute averages and standard deviations for class 0 metrics
prec0_mean = np.mean(precision_scores_class0)
prec0_std = np.std(precision_scores_class0)
rec0_mean = np.mean(recall_scores_class0)
rec0_std = np.std(recall_scores_class0)
f1_0_mean = np.mean(f1_scores_class0)
f1_0_std = np.std(f1_scores_class0)

# Compute averages and standard deviations for class 1 metrics
prec1_mean = np.mean(precision_scores_class1)
prec1_std = np.std(precision_scores_class1)
rec1_mean = np.mean(recall_scores_class1)
rec1_std = np.std(recall_scores_class1)
f1_1_mean = np.mean(f1_scores_class1)
f1_1_std = np.std(f1_scores_class1)

print("\nAverage Results:")
print(f"Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")

print("\nClass 0 Metrics:")
print(f"  Precision: {prec0_mean*100:.2f}% ± {prec0_std*100:.2f}%")
print(f"  Recall:    {rec0_mean*100:.2f}% ± {rec0_std*100:.2f}%")
print(f"  F1-score:  {f1_0_mean*100:.2f}% ± {f1_0_std*100:.2f}%")

print("\nClass 1 Metrics:")
print(f"  Precision: {prec1_mean*100:.2f}% ± {prec1_std*100:.2f}%")
print(f"  Recall:    {rec1_mean*100:.2f}% ± {rec1_std*100:.2f}%")
print(f"  F1-score:  {f1_1_mean*100:.2f}% ± {f1_1_std*100:.2f}%")


# #8 partciepnct - 45
#
# # 0    701
# # 1     47
#
# # target
# 0    6686
# 1     410