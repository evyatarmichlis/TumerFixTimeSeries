import statsmodels.api as sm
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from statsmodels.formula.api import ols
from representation_method.utils.data_loader import DataConfig, load_eye_tracking_data

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


config = DataConfig(
    data_path='data/Categorized_Fixation_Data_1_18.csv',
    approach_num=8,
    normalize=True,
    per_slice_target=True,
    participant_id=1
)

df = load_eye_tracking_data(
    data_path=config.data_path,
    approach_num=config.approach_num,
    participant_id=None,
    data_format="legacy"
)
# Count of each class in the 'target' column (0 vs 1)
target_counts = df['target'].value_counts()

# Number of unique trial indices
num_unique_trials = df['TRIAL_INDEX'].nunique()

# Number of unique recording sessions
num_unique_sessions = df['RECORDING_SESSION_LABEL'].nunique()

# Print results
print("Target counts (0 vs 1):")
print(target_counts)
print(f"\nNumber of unique TRIAL_INDEX values: {num_unique_trials}")
print(f"Number of unique RECORDING_SESSION_LABEL values: {num_unique_sessions}")
quit(1)
feature_columns = [
    'Pupil_Size', 'CURRENT_FIX_DURATION', 'relative_x', 'relative_y',
    'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT'
]
explore_data(df,feature_columns)