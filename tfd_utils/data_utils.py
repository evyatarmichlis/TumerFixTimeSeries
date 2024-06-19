from typing import Union, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from tqdm import tqdm

from tfd_utils.logger_utils import print_and_log


def dtype_range(dtype):
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min, np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).min, np.finfo(dtype).max
    elif np.issubdtype(dtype, np.object_):
        return "N/A", "N/A"
    else:
        return "Unknown Type"


def letter_to_num(letter):
    return ord(letter.upper()) - ord('A') + 1


def convert_zone_value(value):
    if value[0].isdigit():
        # If the first character is a digit, assume the format is '6F'
        num_part = value[:-1]
        letter_part = value[-1]
    else:
        # Otherwise, assume the format is 'F6'
        letter_part = value[0]
        num_part = value[1:]

    # Convert the letter to a number if necessary
    letter_num = letter_to_num(letter_part) if not letter_part.isdigit() else int(letter_part)
    # The numeric part is always assumed to be a number, so convert it
    num = int(num_part)

    return letter_num, num


def features_scaler(df, features_to_scale, scaler_type='minmax'):
    if scaler_type == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif scaler_type == 'zscore':
        scaler = StandardScaler()
    elif scaler_type == 'l2':
        scaler = Normalizer(norm='l2')
    else:
        raise ValueError("Invalid scaler type. Choose 'minmax', 'zscore', or 'l2'.")
    print_and_log(f'Normalizing using a {scaler_type} scaler')

    def scale_column(column):
        return scaler.fit_transform(column.values.reshape(-1, 1)).flatten()

    for feature in features_to_scale:
        # Apply scaling per trial ('RECORDING_SESSION_LABEL' & 'TRIAL_INDEX'):
        df[feature] = df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'])[feature].transform(scale_column)
        # df[feature] = df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX',
        #                           'CURRENT_FIX_INDEX'])[feature].transform(scale_column)
        # df[feature] = scale_column(df[feature])  # Apply scaling to the whole feature column

    return df


def update_target_to_before_target(df):
    df = df.copy()
    starts = (df['target'] == 1) & (df['target'].shift(1, fill_value=0) == 0)
    before_start = starts.shift(-1, fill_value=False)
    # df.loc[before_start, 'target'] = True
    df['target'] = before_start
    return df


def update_target_to_after_target(df):
    df = df.copy()
    ends = (df['target'] == 1) & (df['target'].shift(-1, fill_value=0) == 0)
    after_start = ends.shift(1, fill_value=False)
    # df.loc[after_start, 'target'] = True
    df['target'] = after_start
    return df


def mark_surrounding_rows(row, df, ia_x: str = 'CURRENT_FIX_IA_X', ia_y: str = 'CURRENT_FIX_IA_Y', unit: int = 1):
    if row['target']:
        x, y = row[ia_x], row[ia_y]
        recording_session_label, trial_index = row['RECORDING_SESSION_LABEL'], row['TRIAL_INDEX']
        # Mark rows where X and Y are within 'unit' of the current row
        surrounding_mask = ((df[ia_x].between(x - unit, x + unit)) &
                            (df[ia_y].between(y - unit, y + unit)) &
                            ~((df[ia_x] == x) & (df[ia_y] == y)) &
                            (df['RECORDING_SESSION_LABEL'] == recording_session_label) &
                            (df['TRIAL_INDEX'] == trial_index) &
                            (df['target'] == False)
                            )
        # if 'CURRENT_FIX_COMPONENT_IMAGE_NUMBER' in df.keys():
        #     component_image_number = row['CURRENT_FIX_COMPONENT_IMAGE_NUMBER']
        #     surrounding_mask &= (df['CURRENT_FIX_COMPONENT_IMAGE_NUMBER'] == component_image_number)
        df.loc[surrounding_mask, 'to_update'] = True


def get_df_for_training(
        data_file_path: Union[str, List[str]],
        augment: bool = False,
        normalize: bool = False,
        return_bool_asking_if_processed_data: bool = False,
        take_every_x_rows: int = 1,
        change_target_to_before_target: bool = False,
        change_target_to_after_target: bool = False,
        remove_surrounding_to_hits: int = 0,
        update_surrounding_to_hits: int = 0,
        approach_num: int = 0,
):
    nrows = None  # Read all the rows
    # Original keys for the processed data:
    orig_pr_data_keys = ['RECORDING_SESSION_LABEL', 'TRIAL_INDEX', 'CURRENT_FIX_INDEX', 'Pupil_Size',
                         'CURRENT_FIX_DURATION', 'CURRENT_FIX_INTEREST_AREA_LABEL', 'CURRENT_FIX_COMPONENT_COUNT',
                         'CURRENT_FIX_COMPONENT_INDEX', 'CURRENT_FIX_COMPONENT_DURATION', 'Zones', 'Hit']
    # Original keys for the formatted processed data:
    orig_f_pr_data_keys = ['RECORDING_SESSION_LABEL', 'TRIAL_INDEX', 'CURRENT_FIX_COMPONENT_IMAGE_NUMBER',
                           'CURRENT_FIX_COMPONENT_IMAGE_FILE', 'CURRENT_FIX_INDEX', 'Pupil_Size',
                           'CURRENT_FIX_DURATION', 'CURRENT_FIX_INTEREST_AREA_LABEL', 'CURRENT_FIX_COMPONENT_COUNT',
                           'CURRENT_FIX_COMPONENT_INDEX', 'CURRENT_FIX_COMPONENT_DURATION', 'Zones', 'Hit']
    # Original keys for the categorized processed data:
    orig_cat_f_data_keys = ['RECORDING_SESSION_LABEL', 'TRIAL_INDEX', 'CURRENT_FIX_COMPONENT_IMAGE_NUMBER',
                            'CURRENT_FIX_COMPONENT_IMAGE_FILE', 'CURRENT_FIX_INDEX', 'Pupil_Size',
                            'CURRENT_FIX_DURATION', 'CURRENT_FIX_INTEREST_AREA_LABEL', 'CURRENT_FIX_COMPONENT_COUNT',
                            'CURRENT_FIX_COMPONENT_INDEX', 'CURRENT_FIX_COMPONENT_DURATION', 'Zones', 'Hit',
                            'SCAN_TYPE', 'SLICE_TYPE', 'LOCATION_TYPE']
    # Original keys for the raw data:
    orig_raw_data_keys = ['RECORDING_SESSION_LABEL', 'TRIAL_INDEX', 'CURRENT_FIX_INDEX', 'SAMPLE_INDEX',
                          'SAMPLE_START_TIME', 'IN_BLINK', 'IN_SACCADE', 'Pupil_Size', 'TARGET_ZONE', 'TARGET_XY',
                          'GAZE_IA', 'GAZE_XY', 'Hit']

    # unique_in_raw = ['SAMPLE_INDEX', 'SAMPLE_START_TIME', 'IN_BLINK', 'IN_SACCADE',
    #                  'TARGET_ZONE', 'TARGET_XY', 'GAZE_IA', 'GAZE_XY']
    # unique_in_processed = ['CURRENT_FIX_DURATION', 'CURRENT_FIX_INTEREST_AREA_LABEL', 'CURRENT_FIX_COMPONENT_COUNT',
    #                        'CURRENT_FIX_COMPONENT_INDEX', 'CURRENT_FIX_COMPONENT_DURATION', 'Zones']

    if take_every_x_rows <= 0:
        raise ValueError('Can take data with jumps of positive number of rows ONLY')
    if take_every_x_rows != 1:
        print_and_log(f'Taking data with jumps of {take_every_x_rows} rows', logging_type='warning')

    def skip_rows(x):
        return x % take_every_x_rows != 0  # If <take_every_x_rows == 1> then all the data is used

    if isinstance(data_file_path, str):
        data_file_path = [data_file_path]
    dfs = []
    data_file_path_loop = data_file_path if len(data_file_path) == 1 else tqdm(data_file_path, desc='Loading files')
    for data_fp in data_file_path_loop:
        if data_fp.endswith('.xlsx'):
            data_fp_df = pd.read_excel(data_fp, nrows=nrows, skiprows=skip_rows)
        elif data_fp.endswith('.csv'):
            data_fp_df = pd.read_csv(data_fp, nrows=nrows, skiprows=skip_rows)
        else:
            raise ValueError(f'Unsupported file type for {data_fp}')
        dfs.append(data_fp_df)
    df = pd.concat(dfs, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    # Save the original index as a column:
    df['original_index'] = df.index + 2  # +2 because of the header and the 0-based index

    # print_and_log(f'Df len before - {len(df)}')
    # df = df[df['RECORDING_SESSION_LABEL'] == 23]
    # print_and_log(f'Df len after - {len(df)}')

    # for col in df.columns:
    #     print(f"Column: {col}, Type: {df[col].dtype}, Type Min/Max: {dtype_range(df[col].dtype)}, "
    #           f"Value Min/Max: {df[col].min()}/{df[col].max()}")

    for data_key in ('AILMENT_NUMBER', 'Unnamed: 0'):
        if data_key in df.keys():  # Undesired data
            df = df.drop(data_key, axis=1)

    df_keys = sorted(list(df.keys()))
    if 'original_index' in df_keys:
        df_keys.remove('original_index')

    is_pr_data = len(df_keys) == len(orig_pr_data_keys) and sorted(df_keys) == sorted(orig_pr_data_keys)
    is_f_pr_data = len(df_keys) == len(orig_f_pr_data_keys) and sorted(df_keys) == sorted(orig_f_pr_data_keys)
    is_cat_f_pr_data = (len(df_keys) == len(orig_cat_f_data_keys) and
                        sorted(df_keys) == sorted(orig_cat_f_data_keys))
    if is_pr_data or is_f_pr_data or is_cat_f_pr_data:
        processed_data = True
    elif len(df_keys) == len(orig_raw_data_keys) and sorted(df_keys) == sorted(orig_raw_data_keys):
        processed_data = False
    else:
        raise ValueError(f'Given file does not contain supported keys. Supported keys are either:\n'
                         f'{orig_pr_data_keys}\n'
                         f'Or:\n{orig_f_pr_data_keys}\n'
                         f'Or:\n{orig_cat_f_data_keys}\n'
                         f'Or:\n{orig_raw_data_keys}\n'
                         f'Got:\n{df_keys}')

    print_and_log(f'The original xlsx size is ({df.shape[0]} rows) X ({df.shape[1]} columns)')

    # invalid_value, invalid_ia_str_value = 0, '@0'
    invalid_value, invalid_ia_str_value = -1, '?-1'  # Remember, 'Zone' have '0' as the invalid value for this key.
    df = df.replace(to_replace='.', value=invalid_value)
    df = df.replace(to_replace=np.nan, value=invalid_value)

    df.loc[:, 'target'] = (df['Hit'] == 2).astype(bool)  # Categorized data is 2 for nodule, 1 surrounding, 0 non-nodule

    # Keys types conversions:
    df['RECORDING_SESSION_LABEL'] = df['RECORDING_SESSION_LABEL'].astype(np.int8)
    df['TRIAL_INDEX'] = df['TRIAL_INDEX'].astype(np.int8)
    df['CURRENT_FIX_INDEX'] = df['CURRENT_FIX_INDEX'].astype(np.int16)
    df['Pupil_Size'] = df['Pupil_Size'].astype(float).astype(np.int16)
    df['target'] = df['target'].astype(bool)
    if processed_data:
        # Updated '-1' to an (X, Y) value that would be converted to (-1, -1) - the '?' sign would be converted to -1
        df['CURRENT_FIX_INTEREST_AREA_LABEL'] = df['CURRENT_FIX_INTEREST_AREA_LABEL'].replace(
            to_replace=str(invalid_value), value=invalid_ia_str_value)
        df['CURRENT_FIX_INTEREST_AREA_LABEL'] = df['CURRENT_FIX_INTEREST_AREA_LABEL'].replace(
            to_replace=invalid_value, value=invalid_ia_str_value)
        # Create two new columns
        df['CURRENT_FIX_IA_X'] = df['CURRENT_FIX_INTEREST_AREA_LABEL'].apply(lambda x: letter_to_num(x[0]))
        df['CURRENT_FIX_IA_Y'] = df['CURRENT_FIX_INTEREST_AREA_LABEL'].apply(lambda x: int(x[1:]))
        df = df.drop('CURRENT_FIX_INTEREST_AREA_LABEL', axis=1)

        # Updating '0' to an (X, Y) value that would be converted to (-1, -1) - the '?' sign would be converted to -1
        df['Zones'] = df['Zones'].replace(to_replace='0', value=invalid_ia_str_value)
        df['Zones'] = df['Zones'].replace(to_replace=0, value=invalid_ia_str_value)
        df['Zones'] = df['Zones'].replace(to_replace=str(invalid_value), value=invalid_ia_str_value)
        df['Zones'] = df['Zones'].replace(to_replace=invalid_value, value=invalid_ia_str_value)
        df['Zones'] = df['Zones'].str.split(',\s*')  # Split the 'Zones' column on comma with arbitrary number of spaces
        df = df.explode('Zones')  # Split the list items into separate rows
        # Create two new columns
        # df['Zones_X'] = df['Zones'].apply(lambda x: letter_to_num(x[0]))
        # df['Zones_Y'] = df['Zones'].apply(lambda x: int(x[1:]))
        df['Zones_X'] = df['Zones'].apply(lambda x: convert_zone_value(x)[0])
        df['Zones_Y'] = df['Zones'].apply(lambda x: convert_zone_value(x)[1])
        df = df.drop(labels='Zones', axis=1)

        # Keys types conversions:
        df['CURRENT_FIX_DURATION'] = df['CURRENT_FIX_DURATION'].astype(np.int32)
        df['CURRENT_FIX_IA_X'] = df['CURRENT_FIX_IA_X'].astype(np.int8)
        df['CURRENT_FIX_IA_Y'] = df['CURRENT_FIX_IA_Y'].astype(np.int8)
        df['CURRENT_FIX_COMPONENT_COUNT'] = df['CURRENT_FIX_COMPONENT_COUNT'].astype(np.int16)
        df['CURRENT_FIX_COMPONENT_INDEX'] = df['CURRENT_FIX_COMPONENT_INDEX'].astype(np.int16)
        df['CURRENT_FIX_COMPONENT_DURATION'] = df['CURRENT_FIX_COMPONENT_DURATION'].astype(np.int16)
        df['Zones_X'] = df['Zones_X'].astype(np.int8)
        df['Zones_Y'] = df['Zones_Y'].astype(np.int8)
        if 'CURRENT_FIX_COMPONENT_IMAGE_NUMBER' in df.keys():
            df['CURRENT_FIX_COMPONENT_IMAGE_NUMBER'] = df['CURRENT_FIX_COMPONENT_IMAGE_NUMBER'].astype(np.int16)
        # for drop_key in ['CURRENT_FIX_COMPONENT_IMAGE_FILE']:  # 'SCAN_TYPE', 'SLICE_TYPE', 'LOCATION_TYPE'
        #     if drop_key in df.keys():
        #         df = df.drop(labels=drop_key, axis=1)
    else:  # Raw data
        # Updated '-1' to an (X, Y) value that would be converted to (-1, -1) - the '?' sign would be converted to -1
        df['GAZE_IA'] = df['GAZE_IA'].replace(to_replace=str(invalid_value), value=invalid_ia_str_value)
        df['GAZE_IA'] = df['GAZE_IA'].replace(to_replace=invalid_value, value=invalid_ia_str_value)
        # Create two new columns
        df['GAZE_IA_X'] = df['GAZE_IA'].apply(lambda x: letter_to_num(x[0]))
        df['GAZE_IA_Y'] = df['GAZE_IA'].apply(lambda x: int(x[1:]))
        df = df.drop('GAZE_IA', axis=1)

        # Keys types conversions:
        df['GAZE_IA_X'] = df['GAZE_IA_X'].astype(np.int8)
        df['GAZE_IA_Y'] = df['GAZE_IA_Y'].astype(np.int8)
        df['SAMPLE_INDEX'] = df['SAMPLE_INDEX'].astype(np.int16)
        df['SAMPLE_START_TIME'] = df['SAMPLE_START_TIME'].astype(np.int32)
        df['IN_BLINK'] = df['IN_BLINK'].astype(bool)
        df['IN_SACCADE'] = df['IN_SACCADE'].astype(bool)

        # These three parameters will not be used in training:
        df = df.drop('TARGET_ZONE', axis=1)
        df = df.drop('TARGET_XY', axis=1)
        df = df.drop('GAZE_XY', axis=1)

    for key in df.keys():
        if key in ('CURRENT_FIX_COMPONENT_IMAGE_FILE', 'SCAN_TYPE', 'SLICE_TYPE', 'LOCATION_TYPE'):
            continue
        if ((df[key] < 0) & (df[key] != invalid_value)).any():
            raise ValueError(f'Df at key <{key}> contains negative non invalid_value ({invalid_value}) values')

    if normalize:  # Normalize data points used in training:
        print_and_log('Normalizing data points used in training')
        if processed_data:
            features_to_scale = [
                'CURRENT_FIX_INDEX',
                'Pupil_Size',
                'CURRENT_FIX_DURATION',
                'CURRENT_FIX_COMPONENT_COUNT',
                'CURRENT_FIX_COMPONENT_DURATION',
                # 'CURRENT_FIX_IA_X',
                # 'CURRENT_FIX_IA_Y'
            ]
            # Unused features:
            #   'RECORDING_SESSION_LABEL' - Not used in training
            #   'TRIAL_INDEX' - Not used in training
            #   'Zones' - Not used in training
            #   'Hit'- Boolean
            #   'CURRENT_FIX_COMPONENT_INDEX' - Because this parameter values range is very small
            #   'CURRENT_FIX_COMPONENT_IMAGE_NUMBER'
        else:
            features_to_scale = ['CURRENT_FIX_INDEX',
                                 'Pupil_Size',
                                 'SAMPLE_INDEX',
                                 'SAMPLE_START_TIME']
            # Unused features:
            #   'RECORDING_SESSION_LABEL' - Not used in training
            #   'TRIAL_INDEX' - Not used in training
            #   'TARGET_ZONE' - Not used in training
            #   'TARGET_XY' - Not used in training
            #   'GAZE_XY' - Not used in training
            #   'Hit' - Boolean
            #   'IN_BLINK' - Boolean
            #   'IN_SACCADE' - Boolean
            #   'GAZE_IA_X' - Because this parameter values range only from 1 to 12
            #   'GAZE_IA_Y' - Because this parameter values range only from 1 to 12
            # TODO: Try with 'GAZE_IA_X', 'GAZE_IA_Y'
        print_and_log(f'Normalizing this features:\n{features_to_scale}')
        scaler_type = 'minmax'
        # scaler_type = 'zscore'
        # scaler_type = 'l2'
        df = features_scaler(df=df, features_to_scale=features_to_scale, scaler_type=scaler_type)

    if augment:
        print_and_log('Augmenting data')
        pos_inds = df['Pupil_Size'] > 0
        df.loc[pos_inds, 'Pupil_Size'] += np.random.randint(-10, 11, df.loc[pos_inds, 'Pupil_Size'].shape[0])

        # df_tmp = pd.DataFrame()
        #
        # directions = list(itertools.product([-1, 0, 1], repeat=2))
        # directions.remove((0, 0))
        #
        # for dx, dy in directions:
        #     df_copy = df.copy()
        #     df_copy['CURRENT_FIX_IA_X'] += dx
        #     df_copy['CURRENT_FIX_IA_Y'] += dy
        #     df_tmp = pd.concat([df_tmp, df_copy], ignore_index=True)
        # df = pd.concat([df, df_tmp], ignore_index=True)

    # Clean interest areas point out of the grid:
    print_and_log(f'Len of df before cleanup of out of the 12X12 grid interest areas - {len(df)}')
    ia_x = 'CURRENT_FIX_IA_X' if processed_data else 'GAZE_IA_X'
    ia_y = 'CURRENT_FIX_IA_Y' if processed_data else 'GAZE_IA_Y'
    df = df[(df[ia_x] >= 0) & (df[ia_x] <= 12)]
    df = df[(df[ia_y] >= 0) & (df[ia_y] <= 12)]
    df = df.reset_index(drop=True)  # Reset the index after the cleanup, important for data splits
    print_and_log(f'Len df after - {len(df)}')

    # # For data viewing
    # for key in df.keys():
    #     print_and_log(f'For key {key} got uniques - {sorted(set(df[key]))}')

    # Analyze the distribution of the target variable ('Hit'):
    print_and_log('The distribution of the target variable (Hit):')
    print_and_log(df['target'].value_counts(normalize=True))
    print_and_log(df['target'].value_counts(normalize=False))

    # df = df.replace(-1, np.nan)

    # # Reduce <target = false> data points to the size of the <target = true> data points for balance:
    # print(f'Down-sampling target == False:\nSize before was {len(df)}')
    # true_df = df[df['target']]
    # false_df = df[~df['target']]
    # sampled_false_df = false_df.sample(n=len(true_df))
    # df = pd.concat([true_df, sampled_false_df]).reset_index(drop=True)  # To shuffle - df.sample(frac=1)
    # print(f'Size after is {len(df)}')

    # # Good for ML/DL, convert categorical data to dummy-data / indicator-data
    # df = pd.get_dummies(df, columns='CURRENT_FIX_INTEREST_AREA_LABEL')

    if change_target_to_before_target:
        # Update the preceding data point to the hit(s) to be the point of interest
        df = update_target_to_before_target(df=df)

    if change_target_to_after_target:
        # Update the succeeding data point to the hit(s) to be the point of interest
        df = update_target_to_after_target(df)

    if remove_surrounding_to_hits or update_surrounding_to_hits:
        if remove_surrounding_to_hits and update_surrounding_to_hits:
            raise ValueError('Can not remove and update surrounding to hits at the same time')

        ia_x, ia_y = ('CURRENT_FIX_IA_X', 'CURRENT_FIX_IA_Y') if processed_data else ('GAZE_IA_X', 'GAZE_IA_Y')
        df['to_update'] = False
        df.apply(lambda row: mark_surrounding_rows(row=row,
                                                   df=df,
                                                   ia_x=ia_x,
                                                   ia_y=ia_y,
                                                   unit=remove_surrounding_to_hits or update_surrounding_to_hits),
                 axis=1)

        print_and_log(f'Sum of values in the <to_update> column that are True - {df["to_update"].sum()}')
        if update_surrounding_to_hits:
            print_and_log('Updating to <True> rows surrounding the rows where <target == True>')
            df.loc[df['to_update'], 'target'] = True
        else:
            print_and_log('Removing rows surrounding the rows where <target == True>')
            print_and_log(f'Df length before removal of rows surrounding hits rows - {len(df)}')
            df = df[~df['to_update']]
            print_and_log(f'Df length after - {len(df)}')
        df.drop('to_update', axis=1, inplace=True)

    # print_and_log("Removing rows where SLICE_TYPE == 'ABNORMAL_SLICE'")
    # print_and_log(f'Len df before removal of rows where SLICE_TYPE == "ABNORMAL_SLICE" - {len(df)}')
    # df = df[df['SLICE_TYPE'] != 'ABNORMAL_SLICE']
    # print_and_log(f'Len df after - {len(df)}')

    # print_and_log("Removing rows where SLICE_TYPE == 'NODULE_SLICE':")
    # print_and_log(f'Df len before - {len(df)}')
    # df = df[df['SLICE_TYPE'] != 'NODULE_SLICE']
    # print_and_log(f'Len df after - {len(df)}')

    # print_and_log(f'Number of targets before updating non-normal slices to targets is {df["target"].sum()}')
    # df.loc[df['SCAN_TYPE'] != 'NORMAL', 'target'] = True
    # print_and_log(f'Number of targets after updating non-normal slices to targets is {df["target"].sum()}')

    # print_and_log('Take only the first participant:')
    # print_and_log(f'Df len before - {len(df)}')
    # df = df[df['RECORDING_SESSION_LABEL'] == 1]
    # print_and_log(f'Df len after - {len(df)}')

    # print_and_log('Take only half of the normal data rows:')
    # print_and_log(f'Df len before - {len(df)}')
    # normal_scans = df[df['SCAN_TYPE'] == 'NORMAL'].iloc[::2]
    # other_scans = df[df['SCAN_TYPE'] != 'NORMAL']
    # df = pd.concat([normal_scans, other_scans])
    # df = df.sort_index()
    # print_and_log(f'Df len after - {len(df)}')

    # >>> set(df['SCAN_TYPE'])
    # {'ABNORMAL', 'NORMAL'}
    # >>> set(df['SLICE_TYPE'])
    # {'NORMAL_SLICE', 'ABNORMAL_SLICE', 'NODULE_SLICE'}
    # >>> set(df['LOCATION_TYPE'])
    # {'NORMAL_MISS', 'NODULE_SURROUND', 'NODULE_HIT', 'ABNORMAL_MISS', 'NODULE_MISS'}
    if approach_num <= 0:
        print_and_log(f'No approach (Given approach number {approach_num}). No changes to the data')
    elif approach_num in (1, 2, 3, 4, 5):
        raise ValueError(f'Unsupported approach - {approach_num}. Scan Type prediction is not supported on an ML model')
    elif approach_num == 6:
        print_and_log('==================================\n'
                      'Approach 6.\n'
                      'Include - normal slices,\n'
                      '          abnormal slices,\n'
                      '          non-hit nodule slices\n'
                      'Exclude - none\n'
                      'Prediction level - slice types\n'
                      'Prediction target - nodule slices\n'
                      '==================================\n')
        print_and_log(f'Number of targets before updating targets to be the nodule-slice - {df["target"].sum()}')
        df['target'] = np.where(df['SLICE_TYPE'] == 'NODULE_SLICE', True, False)
        print_and_log(f'Len targets after - {df["target"].sum()}')
    elif approach_num == 7:
        print_and_log('=================================\n'
                      'Approach 7.\n'
                      'Include - normal slices,\n'
                      '          nodule slices\n'
                      'Exclude - abnormal slices\n'
                      'Prediction level - slice type\n'
                      'Prediction target - nodule slice\n'
                      '=================================\n')
        print_and_log(f'Len df before removal of rows where SLICE_TYPE == "ABNORMAL_SLICE" - {len(df)}')
        df = df[df['SLICE_TYPE'] != 'ABNORMAL_SLICE']
        print_and_log(f'Len df after - {len(df)}')

        print_and_log(f'Number of targets before updating targets to be the nodule-slice - {df["target"].sum()}')
        df['target'] = np.where(df['SLICE_TYPE'] == 'NODULE_SLICE', True, False)
        print_and_log(f'Len targets after - {df["target"].sum()}')
    elif approach_num == 8:
        print_and_log('====================================\n'
                      'Approach 8.\n'
                      'Include - normal miss zone,\n'
                      '          abnormal miss zone,\n'
                      '          nodule miss zone,\n'
                      '          nodule surround zone,\n'
                      '          nodule hit zone\n'
                      'Exclude - none\n'
                      'Prediction level - zone type\n'
                      'Prediction target - nodule hit zone\n'
                      '====================================\n')
        print_and_log(f'Number of targets before updating targets to be the nodule-hit-zone - {df["target"].sum()}')
        df['target'] = np.where(df['LOCATION_TYPE'] == 'NODULE_HIT', True, False)
        print_and_log(f'Len targets after - {df["target"].sum()}')
    elif approach_num == 9:
        print_and_log('====================================\n'
                      'Approach 9.\n'
                      'Include - normal miss zone,\n'
                      '          nodule hit zone\n'
                      'Exclude - abnormal miss zone,\n'
                      '          nodule miss zone,\n'
                      '          nodule surround zone\n'
                      'Prediction level - zone type\n'
                      'Prediction target - nodule hit zone\n'
                      '====================================')
        print_and_log(f"Len df before removal of rows where LOCATION_TYPE in "
                      f"('ABNORMAL_MISS', 'NODULE_MISS', 'NODULE_SURROUND') - {len(df)}")
        df = df[~df['LOCATION_TYPE'].isin(('ABNORMAL_MISS', 'NODULE_MISS', 'NODULE_SURROUND'))]
        print_and_log(f'Len df after - {len(df)}')

        print_and_log(f'Number of targets before updating targets to be the nodule-hit-zone - {df["target"].sum()}')
        df['target'] = np.where(df['LOCATION_TYPE'] == 'NODULE_HIT', True, False)
        print_and_log(f'Len targets after - {df["target"].sum()}')
    else:
        raise ValueError(f'Unsupported approach - {approach_num}')

    # Reset the index after all the changes:
    df.reset_index(drop=True, inplace=True)


    if return_bool_asking_if_processed_data:
        return df, processed_data
    return df





