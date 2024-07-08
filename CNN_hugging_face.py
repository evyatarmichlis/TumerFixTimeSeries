import os

import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import pandas as pd
from sklearn.model_selection import train_test_split

from numpy.lib.stride_tricks import sliding_window_view
from sklearn.utils import class_weight

from data_process import IdentSubRec







TRAIN = True




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
feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X', 'CURRENT_FIX_IA_Y',
                   'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT']

@keras.saving.register_keras_serializable()
class F1Metric(keras.metrics.Metric):
    def __init__(self, name='f1_metric', **kwargs):
        super(F1Metric, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1, output_type=tf.int32)
        y_true = tf.cast(y_true, tf.int32)

        tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
        fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
        fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        p = self.tp / (self.tp + self.fp + keras.backend.epsilon())
        r = self.tp / (self.tp + self.fn + keras.backend.epsilon())
        f1 = 2 * p * r / (p + r + keras.backend.epsilon())
        return f1

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

def make_model_conv_lstm(input_shape, num_classes):
    input_layer = keras.layers.Input(input_shape)

    # Convolutional layers with Dropout and L2 Regularization
    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", kernel_regularizer=keras.regularizers.l2(0.001))(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    conv1 = keras.layers.Dropout(0.3)(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", kernel_regularizer=keras.regularizers.l2(0.001))(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    conv2 = keras.layers.Dropout(0.3)(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same", kernel_regularizer=keras.regularizers.l2(0.001))(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)
    conv3 = keras.layers.Dropout(0.3)(conv3)

    # LSTM layers
    lstm_out = keras.layers.LSTM(units=64, return_sequences=True)(conv3)
    lstm_out = keras.layers.LSTM(units=64)(lstm_out)
    lstm_out = keras.layers.Dropout(0.3)(lstm_out)

    # Fully connected layer with L2 Regularization
    dense = keras.layers.Dense(100, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(lstm_out)
    dense = keras.layers.Dropout(0.3)(dense)
    output_layer = keras.layers.Dense(num_classes, activation="softmax")(dense)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

def make_model(input_shape, num_classes):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)
def create_windows(X, Y, window_size):
    X_windows = []
    Y_windows = []

    for i in range(len(X) - window_size + 1):
        X_window = X[i:i + window_size]
        Y_window = Y[i + window_size - 1]  # Use the last target in the window as the target for the window
        X_windows.append(X_window)
        Y_windows.append(Y_window)

    return np.array(X_windows), np.array(Y_windows)
def split_train_test(time_series_df):
    groups = list(time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']).groups.keys())
    np.random.shuffle(groups)
    test_size = int(len(groups) * 0.2)

    test_groups = groups[:test_size]
    train_groups = groups[test_size:]

    train_df = time_series_df[time_series_df.set_index(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']).index.isin(train_groups)].reset_index(drop=True)
    test_df = time_series_df[time_series_df.set_index(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']).index.isin(test_groups)].reset_index(drop=True)

    return train_df, test_df

def create_time_series(time_series_df, interval='1s',  window_size = 5 ):
    time_series_df = time_series_df.copy()

    time_series_df['Cumulative_Time_Update'] = (time_series_df['CURRENT_FIX_COMPONENT_INDEX'] == 1).astype(int)
    time_series_df['Cumulative_Time'] = 0.0  # Ensure float type for cumulative times

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

    # Set 'Cumulative_Time' as a timedelta for easier resampling
    time_series_df['Cumulative_Time'] = pd.to_timedelta(time_series_df['Cumulative_Time'], unit='s')


    resampled_data = []

    # Process each group separately
    for (session, trial), group in time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']):
        # Set 'Cumulative_Time' as the index for resampling
        group = group.set_index('Cumulative_Time')

        # Resample features to a fixed interval (e.g., 1 second)
        resampled_features = group[feature_columns].resample(interval).mean()

        # Resample the target column using the max function
        resampled_target = group['target'].resample(interval).max()

        # Combine the resampled features and target
        resampled_group = resampled_features.merge(resampled_target, on='Cumulative_Time', how='left')

        # Fill missing values using forward fill and backward fill
        resampled_group.ffill(inplace=True)
        resampled_group.bfill(inplace=True)

        # Add 'RECORDING_SESSION_LABEL' and 'TRIAL_INDEX' back to the resampled group
        resampled_group['RECORDING_SESSION_LABEL'] = session
        resampled_group['TRIAL_INDEX'] = trial

        # Append the resampled group to the list
        resampled_data.append(resampled_group)

    # Concatenate all resampled groups into a single DataFrame
    resampled_df = pd.concat(resampled_data).reset_index()

    # Group by 'RECORDING_SESSION_LABEL' and 'TRIAL_INDEX'
    grouped = resampled_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'])


    samples = []
    labels = []

    for name, group in grouped:
        group = group.sort_values(by='Cumulative_Time')

        for start in range(0, len(group) - window_size + 1):
            window = group.iloc[start:start + window_size]
            features = window[feature_columns].values
            samples.append(features)
            # Append the target value, using the max value within the window
            labels.append(window['target'].max())

    # Convert lists to numpy arrays
    samples = np.array(samples)
    labels = np.array(labels)
    return samples,labels



def balanced_batch_generator(samples_class_0, labels_class_0, samples_class_1, labels_class_1, batch_size):
    half_batch_size = batch_size // 2

    # Ensure both datasets are repeated and shuffled
    dataset_class_0 = tf.data.Dataset.from_tensor_slices((samples_class_0, labels_class_0)).shuffle(len(samples_class_0)).repeat()
    dataset_class_1 = tf.data.Dataset.from_tensor_slices((samples_class_1, labels_class_1)).shuffle(len(samples_class_1)).repeat()

    # Batch the datasets
    dataset_class_0 = dataset_class_0.batch(half_batch_size)
    dataset_class_1 = dataset_class_1.batch(half_batch_size)

    # Zip the datasets together
    balanced_dataset = tf.data.Dataset.zip((dataset_class_0, dataset_class_1))

    # Function to interleave the datasets and ensure the shape is correct
    def interleave_fn(ds_0, ds_1):
        combined_samples = tf.concat([ds_0[0], ds_1[0]], axis=0)
        combined_labels = tf.concat([ds_0[1], ds_1[1]], axis=0)
        return combined_samples, combined_labels

    # Interleave, shuffle, and batch the dataset
    balanced_dataset = balanced_dataset.map(interleave_fn).unbatch().batch(batch_size).shuffle(buffer_size=1000).prefetch(tf.data.AUTOTUNE)

    return balanced_dataset



def oversample_generator(samples_class_0, labels_class_0, samples_class_1, labels_class_1, batch_size):
    half_batch_size = batch_size // 2

    # Function to sample from the majority class to balance with the minority class
    def generator():
        while True:
            # Randomly sample from the majority class
            indices_majority = np.random.choice(len(samples_class_1), size=len(samples_class_0), replace=True)
            sampled_majority_samples = samples_class_1[indices_majority]
            sampled_majority_labels = labels_class_1[indices_majority]

            # Combine the minority class samples with the sampled majority class samples
            combined_samples = np.vstack((samples_class_0, sampled_majority_samples))
            combined_labels = np.hstack((labels_class_0, sampled_majority_labels))

            # Shuffle the combined dataset
            shuffled_indices = np.random.permutation(len(combined_labels))
            combined_samples = combined_samples[shuffled_indices]
            combined_labels = combined_labels[shuffled_indices]

            # Yield batches of the specified size
            for start in range(0, len(combined_labels), batch_size):
                end = start + batch_size
                yield combined_samples[start:end], combined_labels[start:end]

    # Create a TensorFlow dataset from the generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, samples_class_0.shape[1], samples_class_0.shape[2]), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    return dataset


def main(df,window_size= 5):

    interval='10ms'
    train_df, test_df = split_train_test(df)
    X_train, Y_train = create_time_series(train_df, interval,window_size=window_size)
    X_test, Y_test = create_time_series(test_df, interval,window_size=window_size)
    input_shape = (window_size, 6)
    num_classes = 2

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(Y_train), y=Y_train)

    # Convert to a dictionary format required by Keras
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(class_weights_dict)
    model = make_model(input_shape=input_shape, num_classes=num_classes)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )
    epochs = 500

    if TRAIN:

        #test with imbalance oversampling
        class_0_indices = np.where(Y_train == 0)[0]
        class_1_indices = np.where(Y_train == 1)[0]

        samples_class_0 = X_train[class_0_indices]
        labels_class_0 = Y_train[class_0_indices]

        samples_class_1 = X_train[class_1_indices]
        labels_class_1 = Y_train[class_1_indices]
        batch_size = 32
        balanced_train_dataset = oversample_generator(samples_class_0, labels_class_0, samples_class_1,
                                                          labels_class_1, batch_size)

        steps_per_epoch = max(len(samples_class_0), len(samples_class_1)) * 2 // batch_size

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
        ]

        history = model.fit(
            balanced_train_dataset,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,  # Define steps per epoch based on balanced dataset size
            validation_data=(X_test, Y_test),  # Use the original distribution for validation
            callbacks=callbacks,
            verbose=1
        )
        #normal use
        # history = model.fit(
        #     X_train,
        #     Y_train,
        #     batch_size=batch_size,
        #     epochs=epochs,
        #     callbacks=callbacks,
        #     validation_split=0.2,
        #     verbose=1,
        #     class_weight = class_weights_dict
        # )

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    precision_weighted = precision_score(Y_test, y_pred_classes, average='weighted')
    recall_weighted = recall_score(Y_test, y_pred_classes, average='weighted')
    f1_weighted = f1_score(Y_test, y_pred_classes, average='weighted')

    precision_macro = precision_score(Y_test, y_pred_classes, average='macro')
    recall_macro = recall_score(Y_test, y_pred_classes, average='macro')
    f1_macro = f1_score(Y_test, y_pred_classes, average='macro')

    precision_micro = precision_score(Y_test, y_pred_classes, average='micro')
    recall_micro = recall_score(Y_test, y_pred_classes, average='micro')
    f1_micro = f1_score(Y_test, y_pred_classes, average='micro')

    print(f"Weighted Precision: {precision_weighted}")
    print(f"Weighted Recall: {recall_weighted}")
    print(f"Weighted F1-score: {f1_weighted}")

    print(f"Macro Precision: {precision_macro}")
    print(f"Macro Recall: {recall_macro}")
    print(f"Macro F1-score: {f1_macro}")

    print(f"Micro Precision: {precision_micro}")
    print(f"Micro Recall: {recall_micro}")
    print(f"Micro F1-score: {f1_micro}")




if __name__ == '__main__':


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
    df = ident_sub_rec.df
    df['target'] = df['SLICE_TYPE'].apply(lambda x: 1 if x == 'NODULE_SLICE' else 0)
    window = 15
    main(df,window_size=window)


    #first results without adding class balance to the model:
    #Weighted Precision: 0.9716223333336166
    # Weighted Recall: 0.9608698687517453
    # Weighted F1-score: 0.9662139812876347
    # Macro Precision: 0.4934640121992207
    # Macro Recall: 0.4885411571445869
    # Macro F1-score: 0.49091236859086845
    # Micro Precision: 0.9608698687517453
    # Micro Recall: 0.9608698687517453
    # Micro F1-score: 0.9608698687517453

    #second try withadding class balance
    # window size = 5
    # Weighted Precision: 0.9438468040422491
    # Weighted Recall: 0.9145134025597682
    # Weighted F1-score: 0.9287729082501983
    # Macro Precision: 0.5002833273614268
    # Macro Recall: 0.5005694541857866
    # Macro F1-score: 0.4975241883515054
    # Micro Precision: 0.9145134025597682
    # Micro Recall: 0.9145134025597682
    # Micro F1-score: 0.9145134025597682


    # window size = 10 - 100 epochs
    #Weighted Precision: 0.9137752414547997
    # Weighted Recall: 0.8722398402630961
    # Weighted F1-score: 0.8921599296687734
    # Macro Precision: 0.4981799005443969
    # Macro Recall: 0.4965104806123435
    # Macro F1-score: 0.49352666214315266
    # Micro Precision: 0.8722398402630961
    # Micro Recall: 0.8722398402630961
    # Micro F1-score: 0.8722398402630961


    # window size = 15 - 100 epochs
    # Weighted Precision: 0.9275606782876191
    # Weighted Recall: 0.8449985964255637
    # Weighted F1-score: 0.882882959576623
    # Macro Precision: 0.506984739830627
    # Macro Recall: 0.5210768306872349
    # Macro F1-score: 0.49691276052753636
    # Micro Precision: 0.8449985964255637
    # Micro Recall: 0.8449985964255637
    # Micro F1-score: 0.8449985964255637

    # window size = 20 - 100 epochs

    # Weighted
    # Precision: 0.9135915228311815
    # Weighted
    # Recall: 0.8501689685571554
    # Weighted
    # F1 - score: 0.8799632770918373
    # Macro
    # Precision: 0.4991192736863065
    # Macro
    # Recall: 0.4979234465036716
    # Macro
    # F1 - score: 0.4905461616369552
    # Micro
    # Precision: 0.8501689685571554
    # Micro
    # Recall: 0.8501689685571554
    # Micro
    # F1 - score: 0.8501689685571554
    #

    # window size = 30 - 100 epochs

    #     Weighted Precision: 0.8536910631896126
    # Weighted Recall: 0.8645426350344383
    # Weighted F1-score: 0.8590591644341848
    # Macro Precision: 0.4794185713153696
    # Macro Recall: 0.48250325278388895
    # Macro F1-score: 0.480807564569495
    # Micro Precision: 0.8645426350344383
    # Micro Recall: 0.8645426350344383
# Micro F1-score: 0.8645426350344383





# test with f1 as loss- window - 15
# Weighted Precision: 0.9496710234810042
# Weighted Recall: 0.8972477791878173
# Weighted F1-score: 0.922542137528438
# Macro Precision: 0.4923708072395619
# Macro Recall: 0.4771235150084106
# Macro F1-score: 0.48124493454866185
# Micro Precision: 0.8972477791878173
# Micro Recall: 0.8972477791878173
# Micro F1-score: 0.8972477791878173

#res with lstm model
# Weighted Precision: 0.9596333843520011
# Weighted Recall: 0.6981725351016469
# Weighted F1-score: 0.8052638723958315
# Macro Precision: 0.4986145834057612
# Macro Recall: 0.48550146157023477
# Macro F1-score: 0.4276079437728776
# Micro Precision: 0.6981725351016469
# Micro Recall: 0.6981725351016469
# Micro F1-score: 0.6981725351016469


#only minorty
# Weighted Precision: 0.9922589351490567
# Weighted Recall: 0.8769739810497819
# Weighted F1-score: 0.9309680700207712
# Macro Precision: 0.4987382254430667
# Macro Recall: 0.46423159071673653
# Macro F1-score: 0.468683961620218
# Micro Precision: 0.876973981049782
# Micro Recall: 0.876973981049782
# Micro F1-score: 0.876973981049782


#oversampling
# Weighted Precision: 0.9803214206444
# Weighted Recall: 0.9691575935491064
# Weighted F1-score: 0.9747037935719062
# Macro Precision: 0.49588844325108683
# Macro Recall: 0.4912548430603934
# Macro F1-score: 0.4933702567247812
# Micro Precision: 0.9691575935491064
# Micro Recall: 0.9691575935491064
# Micro F1-score: 0.9691575935491064