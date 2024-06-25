
import matplotlib.cm as cm
import os

import matplotlib.colors as mcolors
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, make_scorer, f1_score

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV
import lightgbm as lgb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_process import IdentSubRec
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sktime.transformations.panel.rocket import (
    MiniRocket,
    MiniRocketMultivariate,
    MiniRocketMultivariateVariable,
)


input_data_points = [
    'CURRENT_FIX_INDEX',
    'Pupil_Size',
    'CURRENT_FIX_DURATION',
    'CURRENT_FIX_IA_X',
    'CURRENT_FIX_IA_Y',
    'CURRENT_FIX_COMPONENT_COUNT',
    'CURRENT_FIX_COMPONENT_INDEX',
    'CURRENT_FIX_COMPONENT_DURATION',
]


def generate_intervals():
    intervals = []
    # Generate intervals from 10ms to 990ms with a step of 10ms
    for i in np.arange(10, 110, 10):
        intervals.append(f'{i}ms')
    # Generate intervals from 1s to 5s with a step of 0.5s
    for i in np.arange(1.0, 5.5, 0.5):
        intervals.append(f'{i:.1f}s')
    return intervals

def create_time_series(time_series_df, interval='1s'):
    time_series_df = time_series_df.copy()

    time_series_df['Cumulative_Time_Update'] = (time_series_df['CURRENT_FIX_COMPONENT_INDEX'] == 1).astype(int)
    time_series_df['Cumulative_Time'] = 0

    # Iterate over each group and calculate cumulative time correctly
    for _, group in time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']):
        cumulative_time = 0
        cumulative_times = []
        for index, row in group.iterrows():
            if row['Cumulative_Time_Update'] == 1:
                cumulative_time += row['CURRENT_FIX_DURATION']
            cumulative_times.append(cumulative_time)
        time_series_df.loc[group.index, 'Cumulative_Time'] = cumulative_times

    time_series_df['Unique_Index'] = time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']).cumcount()
    time_series_df['Cumulative_Time'] += time_series_df['Unique_Index'] * 1e-4

    time_series_df['Cumulative_Time'] = pd.to_timedelta(time_series_df['Cumulative_Time'], unit='s')

    all_targets = []

    grouped = time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'])
    feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X', 'CURRENT_FIX_IA_Y']
    samples = []

    for name, group in grouped:
        group = group.sort_values(by='Cumulative_Time')

        interval_groups = group.set_index('Cumulative_Time').resample(interval)

        for interval_name, interval_group in interval_groups:
            resampled_features = interval_group[feature_columns]
            resampled_target = interval_group['target']
            if len(resampled_features) ==0:
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
    return X,Y
# Filter data around each hit event
def filter_data_around_hits(data, time_window):
    hit_events = data[data['target'] == True]

    mask = pd.Series([False] * len(data))

    for idx, hit_event in hit_events.iterrows():
        session_label = hit_event['RECORDING_SESSION_LABEL']
        trial_index = hit_event['TRIAL_INDEX']
        hit_time = hit_event['CURRENT_FIX_INDEX']

        time_mask = (
                (data['RECORDING_SESSION_LABEL'] == session_label) &
                (data['TRIAL_INDEX'] == trial_index) &
                (data['CURRENT_FIX_INDEX'] >= hit_time - time_window) &
                (data['CURRENT_FIX_INDEX'] <= hit_time + time_window)
        )
        mask = mask | time_mask

    filtered_data = data[mask].copy()

    return filtered_data

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
    intervals = generate_intervals()[::-1]

    def objective(params):
        interval = '10ms'
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

        train_dataset = lgb.Dataset(X_training, label=Y_train)
        test_dataset = lgb.Dataset(X_test, label=Y_test)

        pos_weight = np.sum(Y_train == 0) / np.sum(Y_train == 1)

        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': params['lgb_learning_rate'],
            'num_leaves': int(params['lgb_num_leaves']),
            'min_child_samples': int(params['lgb_min_child_samples']),
            'scale_pos_weight': pos_weight
        }

        lgb_model = lgb.train(lgb_params, train_dataset, valid_sets=[train_dataset, test_dataset])
        lgb_predictions = lgb_model.predict(X_test)
        lgb_binary_predictions = (lgb_predictions > 0.5).astype(int)

        lgb_roc_auc = roc_auc_score(Y_test, lgb_binary_predictions)

        return {'loss': -lgb_roc_auc, 'status': STATUS_OK}

    space = {
        'lgb_learning_rate': hp.uniform('lgb_learning_rate', 0.01, 0.3),
        'lgb_num_leaves': hp.quniform('lgb_num_leaves', 20, 150, 1),
        'lgb_min_child_samples': hp.quniform('lgb_min_child_samples', 5, 50, 1),
    }

    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)

    print("Best parameters: ", best)




def calculate_metrics(Y_test, predictions):
    # Calculate accuracy, recall, and precision
    accuracy = accuracy_score(Y_test, predictions)
    recall = recall_score(Y_test, predictions, average='binary')
    precision = precision_score(Y_test, predictions, average='binary')

    return accuracy, recall, precision

def plot_movements_specific_scans(filtered_data, session_labels, trial_indices, time_window):
    for session_label in session_labels:
        for trial_index in trial_indices:
            session_data = filtered_data[(filtered_data['RECORDING_SESSION_LABEL'] == session_label) &
                                         (filtered_data['TRIAL_INDEX'] == trial_index)]

            if session_data.empty:
                continue

            hit_events = session_data[session_data['target'] == True]

            for idx, hit_event in hit_events.iterrows():
                hit_time = hit_event['CURRENT_FIX_INDEX']
                hit_x = hit_event['CURRENT_FIX_IA_X']
                hit_y = hit_event['CURRENT_FIX_IA_Y']

                # Time window filter
                time_mask = (
                        (session_data['CURRENT_FIX_INDEX'] >= hit_time - time_window) &
                        (session_data['CURRENT_FIX_INDEX'] <= hit_time + time_window)
                )
                plot_data = session_data[time_mask]

                # Plot the movements
                fig, ax = plt.subplots(figsize=(10, 6))

                norm = mcolors.Normalize(vmin=filtered_data['Pupil_Size'].min(), vmax=filtered_data['Pupil_Size'].max())
                cmap = cm.get_cmap('coolwarm')  # Color map for heatmap

                for i in range(len(plot_data) - 1):
                    x = [plot_data.iloc[i]['CURRENT_FIX_IA_X'], plot_data.iloc[i + 1]['CURRENT_FIX_IA_X']]
                    y = [plot_data.iloc[i]['CURRENT_FIX_IA_Y'], plot_data.iloc[i + 1]['CURRENT_FIX_IA_Y']]
                    pupil_size = plot_data.iloc[i]['Pupil_Size']
                    fixation_duration = plot_data.iloc[i]['CURRENT_FIX_DURATION']

                    color = cmap(norm(pupil_size))  # Get color based on pupil size
                    line_width = 1 + (fixation_duration / filtered_data['CURRENT_FIX_DURATION'].max() * 10)  # Normalize and scale line width

                    ax.plot(x, y, color=color, linewidth=line_width)  # Use line width based on fixation duration

                # Mark the hit spot
                ax.plot(hit_x, hit_y, 'o', markersize=10, color='red', label='Hit Spot')

                ax.set_title(f'Session: {session_label}, Trial: {trial_index}, Hit Time: {hit_time}')
                ax.set_xlabel('X Coordinate')
                ax.set_ylabel('Y Coordinate')
                plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label='Pupil Size')
                plt.legend()
                plt.show()


# Plot movements around hits

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

    # Hit in the categorized data are 1 for nodule and 0 for non-nodule.
    formatted_rad_init_kwargs = dict(data_file_path=formatted_rad_s1_s18_file_path,
                                     test_data_file_path=None,
                                     augment=False,
                                     join_train_and_test_data=False,
                                     normalize=True,
                                     remove_surrounding_to_hits=0,
                                     update_surrounding_to_hits=0)

    # Hit in the categorized data are 2 for nodule, 1 for surrounding and 0 for non-nodule.
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
    df = ident_sub_rec.df[:10000]


    rocket_main(df)
    # accuracy, recall, precision = calculate_metrics(Y_test,predictions)
    # print(f'Accuracy {accuracy:.2f}%\n')
    # print( f'Precision {precision:.2f}%\n')
    # print(f'Recall {recall :.2f}%\n')


    # t =1
    # filtered_data = filter_data_around_hits(time_series_df, t)
    #
    # session_labels = [1]
    # trial_indices = [1]


    # plot_movements_specific_scans(filtered_data, session_labels, trial_indices, t)














