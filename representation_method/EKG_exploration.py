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
from statsmodels.formula.api import ols
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
    from sklearn.manifold import TSNE
    from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu, f_oneway, kruskal
    import pandas as pd
    import numpy as np
    import itertools

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
        sns.histplot(df_target_0[feature], color='blue', kde=True, stat="density",
                     label='normal', ax=ax, bins=10, alpha=0.6)
        sns.histplot(df_target_1[feature], color='red', kde=True, stat="density",
                     label='non conscious', ax=ax, bins=10, alpha=0.6)
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
    plt.tight_layout()
    plt.show()

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
                equal_var=False  # Use Welch's t-test which doesn't assume equal variances
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
    for feature in feature_to_test:
        # Identify trails that have both target classes
        trails_with_both_classes = []
        for trail in aggregated_features_norm['TRIAL_INDEX'].unique():
            trail_data = aggregated_features_norm[aggregated_features_norm['TRIAL_INDEX'] == trail]
            if len(trail_data['target'].unique()) == 2:  # Has both 0 and 1
                trails_with_both_classes.append(trail)

        # Skip if no trails have both classes
        if len(trails_with_both_classes) == 0:
            print(f"No trails have both classes for feature {feature}")
            continue

        # Filter data to include only those trails
        filtered_data = aggregated_features_norm[
            aggregated_features_norm['TRIAL_INDEX'].isin(trails_with_both_classes)
        ]

        plt.figure(figsize=(12, 8))

        # Create violin plots for filtered trails
        sns.violinplot(
            x='TRIAL_INDEX',
            y=feature,
            hue='target',
            data=filtered_data,
            palette={0: 'blue', 1: 'red'},
            split=True,
            inner='quart'
        )

        plt.title(f'Distribution of {feature} across Trails (Only Trails with Both Classes)')
        plt.xlabel('Trail Index')
        plt.ylabel(feature)
        plt.legend(title='Target', labels=['Normal', 'Non-conscious'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Print which trails have both classes (for reference)
        print(f"Trails showing both classes for {feature}: {sorted(trails_with_both_classes)}")


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

    # Now use the new 'target_label' column in your boxplot
    for feature in feature_to_test:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='target_label', y=feature, data=aggregated_features_norm,
                    palette={'normal': 'blue', 'non conscious': 'red', 'all conscious': 'green',
                             'fwd conscious': 'orange'},
                    legend=False)
        plt.title(f'Box Plot of {feature} by Target')
        plt.xlabel('Target')
        plt.ylabel(feature)
        plt.show()

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

    custom_palette = {0: 'blue', 1: 'red'}

    for feature in feature_to_test:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='TRIAL_INDEX', y=feature, hue='target', data=filtered_df, palette=custom_palette)
        plt.title(f'Distribution of {feature} by TRIAL_INDEX (Only Trials with Target=1)')
        plt.xlabel('TRIAL_INDEX')
        plt.ylabel(feature)
        plt.legend(title='Target', labels=['normal (0)', 'non conscious (1)'])
        plt.show()

    # from sklearn.decomposition import PCA
    #
    # # Trail-specific separability visualization
    # for trail in sorted(aggregated_features_norm['TRIAL_INDEX'].unique()):
    #     trail_data = aggregated_features_norm[aggregated_features_norm['TRIAL_INDEX'] == trail]
    #
    #     # Skip trails with only one class
    #     if len(trail_data['target'].unique()) < 2:
    #         continue
    #
    #     X = trail_data[feature_to_test].values
    #     y = trail_data['target'].values
    #
    #     # Apply PCA for dimensionality reduction
    #     pca = PCA(n_components=2)
    #     X_pca = pca.fit_transform(X)
    #
    #     # Plot
    #     plt.figure(figsize=(8, 6))
    #     for target in [0, 1]:
    #         mask = y == target
    #         plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
    #                     label=f"Target {target}",
    #                     alpha=0.7,
    #                     c='blue' if target == 0 else 'red')
    #
    #     plt.title(f"Trail {trail}: Class Separability")
    #     plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    #     plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
    #
    # X = aggregated_features_norm[feature_to_test].values
    #
    # tsne = TSNE(n_components=2, random_state=42)
    # tsne_result = tsne.fit_transform(X)
    # aggregated_features_norm['tsne-2d-one'] = tsne_result[:, 0]
    # aggregated_features_norm['tsne-2d-two'] = tsne_result[:, 1]
    #
    # colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'orange'}  # adjust if you have more targets
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
    #
    # # --- PCA Visualization ---
    # pca = PCA(n_components=2, random_state=42)
    # pca_result = pca.fit_transform(X)
    # aggregated_features_norm['pca-one'] = pca_result[:, 0]
    # aggregated_features_norm['pca-two'] = pca_result[:, 1]
    #
    # plt.figure(figsize=(8, 8))
    # for target_value in aggregated_features_norm['target'].unique():
    #     indices = aggregated_features_norm['target'] == target_value
    #     plt.scatter(aggregated_features_norm.loc[indices, 'pca-one'],
    #                 aggregated_features_norm.loc[indices, 'pca-two'],
    #                 c=colors[target_value],
    #                 label=f'target {target_value}',
    #                 alpha=0.6)
    # plt.title("PCA Cluster Plot")
    # plt.xlabel("PCA Component 1")
    # plt.ylabel("PCA Component 2")
    # plt.legend()
    # plt.show()
    #
    # umap_model = umap.UMAP(n_components=2, random_state=42)
    # umap_result = umap_model.fit_transform(X)
    # aggregated_features_norm['umap-one'] = umap_result[:, 0]
    # aggregated_features_norm['umap-two'] = umap_result[:, 1]
    #
    # plt.figure(figsize=(8, 8))
    # for target_value in aggregated_features_norm['target'].unique():
    #     indices = aggregated_features_norm['target'] == target_value
    #     plt.scatter(aggregated_features_norm.loc[indices, 'umap-one'],
    #                 aggregated_features_norm.loc[indices, 'umap-two'],
    #                 c=colors[target_value],
    #                 label=f'target {target_value}',
    #                 alpha=0.6)
    # plt.title("UMAP Cluster Plot")
    # plt.xlabel("UMAP Component 1")
    # plt.ylabel("UMAP Component 2")
    # plt.legend()
    # plt.show()



csv_path = Path(__file__).parent.parent / "EKG data" / "ML_ECG_Data.csv"
df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')
df['target'] = np.where(df['LOCATION_TYPE'] == 'MI_HIT', 1, 0)


df = df[df['RECORDING_SESSION_LABEL'] == 'P001']

# Filter rows: keep only those where CURRENT_FIX_INTEREST_AREA_LABEL is a number between 1 and 15
df = df[df['CURRENT_FIX_INTEREST_AREA_LABEL'].apply(lambda x: str(x).isdigit())]
df['CURRENT_FIX_INTEREST_AREA_LABEL'] = df['CURRENT_FIX_INTEREST_AREA_LABEL'].astype(int)
df = df[df['CURRENT_FIX_INTEREST_AREA_LABEL'].between(1, 15)]

# Function that counts visits per area in each trial:
# It counts how many times each area is visited in the trial.
def add_visit_counts(group):
    counts = group['CURRENT_FIX_INTEREST_AREA_LABEL'].value_counts().to_dict()
    max_visits = max(counts.values()) if counts else 0
    group['VISIT_COUNT'] = group['CURRENT_FIX_INTEREST_AREA_LABEL'].map(counts)
    group['MAX_VISIT'] = max_visits
    return group

# Apply the function groupwise and reset the index without dropping the grouping columns
df = df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'], group_keys=False).apply(add_visit_counts).reset_index()

# Create a wide-format DataFrame that has one column per area (AREA_1, AREA_2, ... AREA_15)
wide_counts = (
    df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX', 'CURRENT_FIX_INTEREST_AREA_LABEL'])
      .size()
      .unstack(fill_value=0)
)
wide_counts = wide_counts.rename(columns=lambda x: f"AREA_{x}")

# Merge the wide-format counts back into the original DataFrame
df = df.merge(wide_counts, on=['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'], how='left')
base_features = ['Pupil_Size', 'CURRENT_FIX_DURATION']
wide_features = [f"AREA_{i}" for i in range(1, 16)]
feature_to_test = base_features
# feature_to_test = ['VISIT_COUNT', 'CURRENT_FIX_DURATION']
explore_data(df, feature_to_test)

#
#
#
#
#
#
#
# df_trail = df[df['TRIAL_INDEX'].isin([ 28,38,81])]
# # explore_data(df_trail, feature_to_test)
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import LeaveOneOut
# from sklearn.metrics import classification_report
#
# # For demonstration, we assume df_trail contains your subdataset and features:
# # 'VISIT_COUNT' and 'CURRENT_FIX_DURATION'
# base_features = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'VISIT_COUNT', 'MAX_VISIT']
# wide_features = [f"AREA_{i}" for i in range(1, 16)]
# features = base_features + wide_features
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
# base_features = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'VISIT_COUNT', 'MAX_VISIT']
# wide_features = [f"AREA_{i}" for i in range(1, 16)]
# features = base_features + wide_features
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
# base_features = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'VISIT_COUNT', 'MAX_VISIT']
# wide_features = [f"AREA_{i}" for i in range(1, 16)]
# features = base_features + wide_features
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
#
# import lightgbm as lgb
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline
# from sklearn.model_selection import StratifiedKFold, cross_val_score
# from sklearn.metrics import classification_report
# import numpy as np
#
# # Define your features
# base_features = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'VISIT_COUNT', 'MAX_VISIT']
# wide_features = [f"AREA_{i}" for i in range(1, 16)]
# features = base_features + wide_features
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
#
# # Optionally, display detailed classification reports for each fold:
# import lightgbm as lgb
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import numpy as np
#
# from sklearn.model_selection import StratifiedGroupKFold
# from collections import Counter
# from imblearn.over_sampling import SMOTE
# import lightgbm as lgb
# import numpy as np
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#
# # Define your features
# base_features = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'VISIT_COUNT', 'MAX_VISIT']
# wide_features = [f"AREA_{i}" for i in range(1, 16)]
# features = base_features + wide_features
#
# # Prepare X and y
# X = df[features].fillna(0)
# y = df['target']
#
# # Define group labels (participant + trial)
# groups = df['RECORDING_SESSION_LABEL'].astype(str) + "_" + df['TRIAL_INDEX'].astype(str)
#
# # Initialize metrics storage
# acc_scores = []
# precision_scores_class0 = []
# recall_scores_class0 = []
# f1_scores_class0 = []
# precision_scores_class1 = []
# recall_scores_class1 = []
# f1_scores_class1 = []
#
# # Stratified K-fold with grouping
# sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
# fold = 1
#
# for train_index, test_index in sgkf.split(X, y, groups):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#     # Apply SMOTE
#     X_train_smote, y_train_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
#
#     # Handle imbalance
#     counter = Counter(y_train)
#     scale_pos_weight = counter[0] / counter[1]
#
#     # Train LightGBM
#     clf = lgb.LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight)
#     clf.fit(X_train_smote, y_train_smote)
#
#     # Predict and evaluate
#     y_pred = clf.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     acc_scores.append(acc)
#
#     prec = precision_score(y_test, y_pred, average=None, labels=[0, 1])
#     rec = recall_score(y_test, y_pred, average=None, labels=[0, 1])
#     f1 = f1_score(y_test, y_pred, average=None, labels=[0, 1])
#
#     precision_scores_class0.append(prec[0])
#     recall_scores_class0.append(rec[0])
#     f1_scores_class0.append(f1[0])
#     precision_scores_class1.append(prec[1])
#     recall_scores_class1.append(rec[1])
#     f1_scores_class1.append(f1[1])
#
#     print(f"Fold {fold} - Accuracy: {acc*100:.2f}%")
#     print(f"    Class 0 - Precision: {prec[0]*100:.2f}%, Recall: {rec[0]*100:.2f}%, F1: {f1[0]*100:.2f}%")
#     print(f"    Class 1 - Precision: {prec[1]*100:.2f}%, Recall: {rec[1]*100:.2f}%, F1: {f1[1]*100:.2f}%")
#     fold += 1
#
# # Overall performance summary
# acc_mean = np.mean(acc_scores)
# acc_std = np.std(acc_scores)
#
# print(f"\nOverall Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")
#
# # Compute averages and standard deviations for class 0 metrics
# prec0_mean = np.mean(precision_scores_class0)
# prec0_std = np.std(precision_scores_class0)
# rec0_mean = np.mean(recall_scores_class0)
# rec0_std = np.std(recall_scores_class0)
# f1_0_mean = np.mean(f1_scores_class0)
# f1_0_std = np.std(f1_scores_class0)
#
# # Compute averages and standard deviations for class 1 metrics
# prec1_mean = np.mean(precision_scores_class1)
# prec1_std = np.std(precision_scores_class1)
# rec1_mean = np.mean(recall_scores_class1)
# rec1_std = np.std(recall_scores_class1)
# f1_1_mean = np.mean(f1_scores_class1)
# f1_1_std = np.std(f1_scores_class1)
#
# print("\nAverage Results:")
# print(f"Accuracy: {acc_mean*100:.2f}% ± {acc_std*100:.2f}%")
#
# print("\nClass 0 Metrics:")
# print(f"  Precision: {prec0_mean*100:.2f}% ± {prec0_std*100:.2f}%")
# print(f"  Recall:    {rec0_mean*100:.2f}% ± {rec0_std*100:.2f}%")
# print(f"  F1-score:  {f1_0_mean*100:.2f}% ± {f1_0_std*100:.2f}%")
#
# print("\nClass 1 Metrics:")
# print(f"  Precision: {prec1_mean*100:.2f}% ± {prec1_std*100:.2f}%")
# print(f"  Recall:    {rec1_mean*100:.2f}% ± {rec1_std*100:.2f}%")
# print(f"  F1-score:  {f1_1_mean*100:.2f}% ± {f1_1_std*100:.2f}%")
# # #8 partciepnct - 45
# #
# # # 0    701
# # # 1     47
# #
# # # target
# # 0    6686
# # 1     410