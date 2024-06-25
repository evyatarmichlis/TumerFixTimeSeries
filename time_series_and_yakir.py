import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split, ShuffleSplit, cross_val_score
import seaborn as sns
from lightgbm import LGBMClassifier, Dataset
from datetime import datetime
from tqdm import tqdm


import os

from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, make_scorer, f1_score

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb


from data_process import IdentSubRec
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sktime.transformations.panel.rocket import (
    MiniRocket,
    MiniRocketMultivariate,
    MiniRocketMultivariateVariable,
)

def generate_intervals():
    intervals = []
    for i in np.arange(10, 110, 10):
        intervals.append(f'{i}ms')
    for i in np.arange(1.0, 5.5, 0.5):
        intervals.append(f'{i:.1f}s')
    return intervals

def create_time_series(time_series_df, interval='1s'):
    time_series_df = time_series_df.copy()

    # Ensure numeric types
    numeric_columns = ['CURRENT_FIX_COMPONENT_INDEX', 'CURRENT_FIX_DURATION']
    time_series_df[numeric_columns] = time_series_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    time_series_df['Cumulative_Time_Update'] = (time_series_df['CURRENT_FIX_COMPONENT_INDEX'] == 1).astype(int)
    time_series_df['Cumulative_Time'] = 0.0

    for name, group in time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']):
        cumulative_time = 0.0
        cumulative_times = []
        for _, row in group.iterrows():
            if row['Cumulative_Time_Update'] == 1:
                cumulative_time += row['CURRENT_FIX_DURATION']
            cumulative_times.append(cumulative_time)
        time_series_df.loc[group.index, 'Cumulative_Time'] = cumulative_times

    time_series_df['Unique_Index'] = time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']).cumcount()
    time_series_df['Cumulative_Time'] += time_series_df['Unique_Index'] * 1e-4

    time_series_df['Cumulative_Time'] = pd.to_timedelta(time_series_df['Cumulative_Time'], unit='s')

    all_targets = []

    grouped = time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'])
    feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X', 'CURRENT_FIX_IA_Y','CURRENT_FIX_INDEX','CURRENT_FIX_COMPONENT_COUNT']
    samples = []

    for name, group in grouped:
        group = group.sort_values(by='Cumulative_Time')

        interval_groups = group.set_index('Cumulative_Time').resample(interval)

        for interval_name, interval_group in interval_groups:
            resampled_features = interval_group[feature_columns]
            resampled_target = interval_group['target']
            if len(resampled_features) == 0:
                continue
            time_series_data = np.array([resampled_features.values])
            time_series_data_shape = time_series_data.shape
            time_series_data = time_series_data.reshape(time_series_data_shape[1], time_series_data_shape[2])
            res = pd.Series([pd.Series(time_series_data[:, i]) for i in range(time_series_data.shape[1])])
            samples.append(res)
            target_value = np.nanmax(resampled_target.values)
            all_targets.append(int(target_value))

    X = pd.DataFrame(samples)
    Y = pd.Series(all_targets)
    return X, Y

def split_train_test(time_series_df):
    groups = list(time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']).groups.keys())
    np.random.shuffle(groups)
    test_size = int(len(groups) * 0.2)

    test_groups = groups[:test_size]
    train_groups = groups[test_size:]

    train_df = time_series_df[time_series_df.set_index(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']).index.isin(train_groups)].reset_index(drop=True)
    test_df = time_series_df[time_series_df.set_index(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']).index.isin(test_groups)].reset_index(drop=True)

    return train_df, test_df


def rocket_main(time_series_df):
    interval = '30ms'  # Choose one interval for this example
    train_df, test_df = split_train_test(time_series_df)
    X_train, Y_train = create_time_series(train_df, interval)
    X_test, Y_test = create_time_series(test_df, interval)

    minirocket_transformer = make_pipeline(
        MiniRocketMultivariateVariable(
            pad_value_short_series=-10.0, random_state=42, max_dilations_per_kernel=16
        ),
        StandardScaler(with_mean=False),
    )

    minirocket_transformer.fit(X_train)
    X_training = minirocket_transformer.transform(X_train)
    X_test = minirocket_transformer.transform(X_test)



def train_and_evaluate(df, test_size=0.2, num_leaves=31, learning_rate=0.1, min_child_samples=20, random_state=42, cross_validate=True, cross_validation_n_splits=10, split_by_participants=False,plot_confusion_matrix=False):
    interval = '30ms'  # Choose one interval for this example
    class_weight = 'balanced'
    clf = LGBMClassifier(
        n_estimators=100,
        class_weight=class_weight,
        random_state=random_state,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        min_child_samples=min_child_samples,
        force_row_wise=True,
        verbose=-1,
    )

    acc_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []
    confusion_matrices = []
    y_preds = []
    y_tests = []

    confusion_matrix_res = np.array([])

    if cross_validate:
        print('Running cross-validation')
        split_indices = []
        unique_participants = df['RECORDING_SESSION_LABEL'].unique()

        if split_by_participants:
            split_test_size = int(len(unique_participants) * test_size)
            np.random.seed(seed=random_state)
            test_participants_labels = np.random.choice(unique_participants, size=(cross_validation_n_splits, split_test_size))
            for test_participants_label in test_participants_labels:
                train_indices = df[~df['RECORDING_SESSION_LABEL'].isin(test_participants_label)].index
                test_indices = df[df['RECORDING_SESSION_LABEL'].isin(test_participants_label)].index
                split_indices.append((train_indices, test_indices))
        else:
            df['group'] = df[['RECORDING_SESSION_LABEL', 'TRIAL_INDEX', 'CURRENT_FIX_INDEX']].apply(
                lambda row: '_'.join(row.values.astype(str)), axis=1)
            gss = GroupShuffleSplit(n_splits=cross_validation_n_splits, test_size=test_size, random_state=random_state)
            for train_idx, test_idx in gss.split(df, groups=df['group']):
                split_indices.append((train_idx, test_idx))

        for split_ind, (train_index, test_index) in tqdm(enumerate(split_indices), total=len(split_indices)):
            train_df = df.iloc[train_index]
            test_df = df.iloc[test_index]

            X_train, Y_train = create_time_series(train_df, interval)
            X_test, Y_test = create_time_series(test_df, interval)

            minirocket_transformer = make_pipeline(
                MiniRocketMultivariateVariable(
                    pad_value_short_series=-10.0, random_state=42, max_dilations_per_kernel=16
                ),
                StandardScaler(with_mean=False),
            )

            minirocket_transformer.fit(X_train)
            X_train_rocket = minirocket_transformer.transform(X_train)
            X_test_rocket = minirocket_transformer.transform(X_test)
            clf.fit(X_train_rocket, Y_train)
            y_pred = clf.predict(X_test_rocket)
            probabilities = clf.predict_proba(X_test_rocket)[:, 1]

            acc_scores.append(accuracy_score(Y_test, y_pred))
            precision_scores.append(precision_score(Y_test, y_pred, average='binary'))
            recall_scores.append(recall_score(Y_test, y_pred, average='binary'))
            f1_scores.append(f1_score(Y_test, y_pred, average='binary'))
            roc_auc_scores.append(roc_auc_score(Y_test, probabilities))
            confusion_matrices.append(confusion_matrix(Y_test, y_pred))
            y_preds.append(y_pred)
            y_tests.append(Y_test)

        confusion_matrix_res = np.mean(confusion_matrices, axis=0).round().astype(int)
        y_pred = np.concatenate(y_preds)
        y_test = np.concatenate(y_tests)
        if plot_confusion_matrix:
            if not cross_validate:
                confusion_matrix_res = confusion_matrix(y_true=y_test, y_pred=y_pred)
        acc = np.mean(acc_scores)
        precision = np.mean(precision_scores)
        recall = np.mean(recall_scores)
        f1 = np.mean(f1_scores)
        roc_auc_score_res = np.mean(roc_auc_scores)
    else:
        train_df, test_df = split_train_test(df)
        X_train, Y_train = create_time_series(train_df, interval)
        X_test, Y_test = create_time_series(test_df, interval)

        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(Y_test, y_pred)
        precision = precision_score(Y_test, y_pred, average='binary')
        recall = recall_score(Y_test, y_pred, average='binary')
        f1 = f1_score(Y_test, y_pred, average='binary')
        roc_auc_score_res = roc_auc_score(Y_test, y_pred)

    print(f'Accuracy {acc * 100:.2f}%\nPrecision {precision * 100:.2f}%\nRecall {recall * 100:.2f}%\nF1 score {f1 * 100:.2f}%\nROC AUC score {roc_auc_score_res * 100:.2f}%')

    if confusion_matrix_res.size > 0:
        title = f'Confusion Matrix. {clf.__class__.__name__} model with class_weight={clf.class_weight}'
        plt.title(title)
        sns.heatmap(confusion_matrix_res, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    return acc, precision, recall, f1, roc_auc_score_res

if __name__ == '__main__':
    new_rad_s1_s9_csv_file_path = 'data/NewRad_Fixations_S1_S9_Data.csv'
    expert_rad_e1_e3_csv_file_path = 'data/ExpertRad_Fixations_E1_E3.csv'
    rad_s1_s18_file_path = 'data/Rad_Fixations_S1_S18_Data.csv'
    formatted_rad_s1_s18_file_path = 'data/Formatted_Fixations_ML.csv'
    categorized_rad_s1_s18_file_path = 'data/Categorized_Fixation_Data_1_18.csv'

    raw_participants_file_paths = ['1_Formatted_Sample.csv', '2_Formatted_Sample.csv', '3_Formatted_Sample.csv',
                                   '4_Formatted_Sample.csv', '6_Formatted_Sample.csv', '7_Formatted_Sample.csv',
                                   '8_Formatted_Sample.csv', '9_Formatted_Sample.csv', '19_Formatted_Sample.csv',
                                   '23_Formatted_Sample.csv', '24_Formatted_Sample.csv', '25_Formatted_Sample.csv',
                                   '30_Formatted_Sample.csv', '33_Formatted_Sample.csv', '34_Formatted_Sample.csv',
                                   '35_Formatted_Sample.csv', '36_Formatted_Sample.csv', '37_Formatted_Sample.csv']
    raw_participants_folder_path = '/media/y/2TB/1_THESIS_FILES/Data_D231110/'
    for i in range(len(raw_participants_file_paths)):
        raw_participants_file_paths[i] = os.path.join(raw_participants_folder_path, raw_participants_file_paths[i])
    raw_participant_1 = '/media/y/2TB/1_THESIS_FILES/Data_D231110/1_Formatted_Sample.csv'

    rad_init_kwargs = dict(data_file_path=rad_s1_s18_file_path,
                           test_data_file_path=None,
                           augment=False,
                           join_train_and_test_data=False,
                           normalize=True)
    raw_participant_1_init_kwargs = dict(data_file_path=raw_participant_1,
                                         test_data_file_path=None,
                                         augment=False,
                                         join_train_and_test_data=False,
                                         normalize=True)
    raw_participants_init_kwargs = dict(data_file_path=raw_participants_file_paths,
                                        test_data_file_path=None,
                                        augment=False,
                                        join_train_and_test_data=False,
                                        normalize=True,
                                        take_every_x_rows=2)
    experts_init_kwargs = dict(data_file_path=expert_rad_e1_e3_csv_file_path,
                               test_data_file_path=None,
                               augment=False,
                               join_train_and_test_data=False,
                               normalize=True)
    experts_and_rad_init_kwargs = dict(data_file_path=[rad_s1_s18_file_path, expert_rad_e1_e3_csv_file_path],
                                       test_data_file_path=None,
                                       augment=False,
                                       join_train_and_test_data=False,
                                       normalize=True)

    formatted_rad_init_kwargs = dict(data_file_path=formatted_rad_s1_s18_file_path,
                                     test_data_file_path=None,
                                     augment=False,
                                     join_train_and_test_data=False,
                                     normalize=True,
                                     remove_surrounding_to_hits=0,
                                     update_surrounding_to_hits=0)

    categorized_rad_init_kwargs = dict(
        data_file_path=categorized_rad_s1_s18_file_path,
        test_data_file_path=None,
        augment=False,
        join_train_and_test_data=False,
        normalize=True,
        remove_surrounding_to_hits=0,
        update_surrounding_to_hits=0,
        approach_num=8,
    )

    ident_sub_rec = IdentSubRec(**categorized_rad_init_kwargs)
    df = ident_sub_rec.df

    acc, precision, recall, f1, roc_auc_score_res = train_and_evaluate(df, cross_validate=True, cross_validation_n_splits=10, split_by_participants=False)
