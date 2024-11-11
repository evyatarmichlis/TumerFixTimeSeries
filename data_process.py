import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
# import torch
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from lightgbm import LGBMClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
# from pytorch_tabnet.pretraining import TabNetPretrainer
# from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report, precision_recall_fscore_support
# from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from tqdm import tqdm

from tfd_utils.data_utils import get_df_for_training, stratified_group_split_independent_unique  # load_config_file
from tfd_utils.logger_utils import print_and_log
from tfd_utils.model_stats_utils import model_stats_per_training_size, model_stats_per_weight_classes, \
    model_stats_per_learning_rate, model_stats_per_num_leaves, model_stats_per_min_child_samples
from tfd_utils.visualization_utils import visualize_data, save_plot_func, visualize_model, compare_old_to_new_data


class SmoteType:
    none = None
    smote = 'smote'
    smote_random_under_sampler = 'smote_random_under_sampler'
    smotetomek = 'smotetomek'
    tomek_links_majority_strategy = 'tomek_links_majority_strategy'
    smoteenn = 'smoteenn'
    adasyn = 'adasyn'


def apply_smote_and_related(x, y, smote_type: Optional[SmoteType] = 'smote', random_state: int = 42,
                            print_once: bool = True):
    if smote_type == SmoteType.smote:
        print_and_log('Using SMOTE', run_once=print_once)
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        x, y = smote.fit_resample(x, y)
    elif smote_type == SmoteType.smote_random_under_sampler:
        print_and_log('Using SMOTE + RandomUnderSampler', run_once=print_once)
        smote = SMOTE(random_state=random_state, k_neighbors=5)
        x, y = smote.fit_resample(x, y)
        under = RandomUnderSampler(sampling_strategy='auto')
        x, y = under.fit_resample(x, y)
    elif smote_type == SmoteType.smotetomek:
        print_and_log('Using SMOTETomek', run_once=print_once)
        tl = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
        x, y = tl.fit_resample(x, y)
    elif smote_type == SmoteType.tomek_links_majority_strategy:
        tl = TomekLinks(sampling_strategy='majority')
        x, y = tl.fit_resample(x, y)
        print_and_log("\nAfter SMOTE:", run_once=print_once)
        print_and_log("TomekLinks of samples in the minority class (after):", sum(y == 1), run_once=print_once)
        print_and_log("TomekLinks of samples in the majority class (after):", sum(y == 0), run_once=print_once)
    elif smote_type == SmoteType.smoteenn:
        print_and_log('Using SMOTEENN', run_once=print_once)
        smoteenn = SMOTEENN(random_state=random_state)
        x, y = smoteenn.fit_resample(x, y)
    elif smote_type == SmoteType.adasyn:
        print_and_log('Using ADASYN', run_once=print_once)
        adasyn = ADASYN(random_state=random_state)
        x, y = adasyn.fit_resample(x, y)
    else:
        raise ValueError(f'Unsupported smote_type - {smote_type}')

    return x, y


class IdentSubRec:
    def __init__(
            self,
            data_file_path: Union[str, List[str]],
            test_data_file_path: Optional[str] = None,
            old_data_file_path: Optional[str] = None,
            augment: bool = False,
            join_train_and_test_data: bool = True,
            normalize: bool = False,
            take_every_x_rows: int = 1,
            change_target_to_before_target: bool = False,
            change_target_to_after_target: bool = False,
            remove_surrounding_to_hits: int = 0,
            update_surrounding_to_hits: int = 0,
            approach_num: int = 0,
    ):
        self.data_file_path = data_file_path
        self.test_data_file_path = test_data_file_path
        self.old_data_file_path = old_data_file_path
        self.approach_num = approach_num

        self.df, self.processed_data, self.gen_data = self.get_df_for_training(
            self.data_file_path,
            augment=augment,
            normalize=normalize,
            return_bool_asking_if_processed_data=True,
            take_every_x_rows=take_every_x_rows,
            change_target_to_before_target=change_target_to_before_target,
            change_target_to_after_target=change_target_to_after_target,
            remove_surrounding_to_hits=remove_surrounding_to_hits,
            update_surrounding_to_hits=update_surrounding_to_hits,
            approach_num=approach_num,
        )

        # # DEBUG! DEBUG! DEBUG!!! For testing - keep only one participant to test on.

        # Only used if a test data file path is provided:
        self.test_df = None if test_data_file_path is None else self.get_df_for_training(self.test_data_file_path,
                                                                                         normalize=normalize)

        if join_train_and_test_data and self.test_df is not None:
            print_and_log('Joining train and test dataframes together and setting the test data to None')
            # Join the two dataframes together and set the test data to None
            self.test_df['RECORDING_SESSION_LABEL'] += 19
            self.test_df['TRIAL_INDEX'] += 42
            self.df = pd.concat([self.df, self.test_df])
            # Reset the index so that it is continuous, important for data splits:
            self.df = self.df.reset_index(drop=True)
            self.test_df = None

        # For old vs new data comparison, only used if an old data file path is provided:
        self.old_df = None if old_data_file_path is None else self.get_df_for_training(self.old_data_file_path,
                                                                                       normalize=normalize)

        # Classifier params:
        self.clf = None
        self.base_clf = None
        # Plot params:
        self.default_font_size = 20
        plt.rcParams.update({'font.size': self.default_font_size})

        print_and_log('Finished loading df data and preparing it for training')

    # Data preparation function:
    @staticmethod
    def get_df_for_training(*args, **kwargs):
        return get_df_for_training(*args, **kwargs)

    # ============================================ Visualization functions ============================================
    @staticmethod
    def save_plot_func(title: str, data: Optional = None):
        return save_plot_func(title=title, data=data)

    def visualize_data(self, start_grid: int = 1, end_grid: int = 12, save_plot_flag: bool = False,
                       run_on_test_data: bool = False):
        return visualize_data(df=self.test_df if run_on_test_data else self.df,
                              data_file_path=self.data_file_path,
                              start_grid=start_grid,
                              end_grid=end_grid,
                              save_plot=save_plot_flag)

    @staticmethod
    def visualize_model(clf):
        return visualize_model(clf=clf)

    @staticmethod
    def compare_old_to_new_data(old_df, df):
        return compare_old_to_new_data(old_df=old_df, df=df)

    # ------------------------------------------ Visualization functions end ------------------------------------------

    # ============================================= Model stats functions =============================================
    def model_stats_per_training_size(self, one_plot: bool = True, save_plot: bool = True):
        return model_stats_per_training_size(ident_sub_rec=self, one_plot=one_plot, save_plot=save_plot)

    def model_stats_per_weight_classes(self, one_plot: bool = True, save_plot: bool = True):
        return model_stats_per_weight_classes(ident_sub_rec=self, one_plot=one_plot, save_plot=save_plot)

    def model_stats_per_learning_rate(self, one_plot: bool = True, save_plot: bool = True,
                                      learning_rates=np.arange(0.01, 0.11, 0.01)):
        return model_stats_per_learning_rate(ident_sub_rec=self, one_plot=one_plot, save_plot=save_plot,
                                             learning_rates=learning_rates)

    def model_stats_per_num_leaves(self, one_plot: bool = True, save_plot: bool = True,
                                   num_leaves_list=np.arange(101, 151, 1)):
        return model_stats_per_num_leaves(ident_sub_rec=self, one_plot=one_plot, save_plot=save_plot,
                                          num_leaves_list=num_leaves_list)

    def model_stats_per_min_child_samples(self, one_plot: bool = True, save_plot: bool = True):
        return model_stats_per_min_child_samples(ident_sub_rec=self, one_plot=one_plot, save_plot=save_plot)

    # ------------------------------------------ Model stats functions end ------------------------------------------

    def train(
            self,
            test_size: float = 0.2,
            class_weight: Optional[dict] = None,
            num_leaves: int = 31,
            learning_rate: float = 0.1,
            min_child_samples: int = 20,
            print_stats: bool = True,
            plot_confusion_matrix: bool = True,
            save_plot: bool = False,
            random_state: int = 0,
            cross_validate: bool = True,
            cross_validation_n_splits: int = 10,
            split_by_participants: bool = False,
            completely_random_data_split: bool = False,
            smote_type: Optional[SmoteType] = False,
            save_models: bool = False,
            save_predicted_as_true_data_rows: bool = False,
            predict_all_train_data: bool = False,
            participants_to_train_on: Optional[List[int]] = None,
            participants_to_test_on: Optional[List[int]] = None,
    ):
        print_and_log(f'Running with num_leaves={num_leaves}, '
                      f'learning_rate={learning_rate}, '
                      f'min_child_samples={min_child_samples}, '
                      f'print_stats={print_stats}, '
                      f'plot_confusion_matrix={plot_confusion_matrix}, '
                      f'save_plot={save_plot}, '
                      f'random_state={random_state}, '
                      f'cross_validate={cross_validate}, '
                      f'cross_validation_n_splits={cross_validation_n_splits}, ',
                      f'split_by_participants={split_by_participants}, '
                      f'completely_random_data_split={completely_random_data_split}, ',
                      f'smote_type={smote_type}, '
                      f'save_models={save_models}, '
                      f'save_predicted_as_true_data_rows={save_predicted_as_true_data_rows}, '
                      f'predict_all_train_data={predict_all_train_data}, ',
                      f'participants_to_train_on={participants_to_train_on}, '
                      f'participants_to_test_on={participants_to_test_on}')
        # ============================================= Init used model: ==============================================
        # self.clf = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
        # self.clf = XGBClassifier()
        # self.clf = LGBMClassifier()
        # self.clf = LGBMClassifier(class_weight='balanced', random_state=random_state)
        # self.clf = SVC()
        # self.clf = SVC(class_weight='balanced')

        # self.clf = LGBMClassifier(class_weight='balanced',
        #                           random_state=random_state,
        #                           num_leaves=31,
        #                           learning_rate=0.1,
        #                           min_child_samples=100)

        if class_weight is None:
            # class_weight = {False: 1, True: 0.2 * (~self.df['target']).sum() / self.df['target'].sum()}
            class_weight = 'balanced'

        # Use a simple PyTorch NN with skorch to have sklearn like usability:
        # from tfd_utils.train_utils.torch_nn import get_skorch_model
        # self.clf = get_skorch_model(input_params=len(input_data_points))

        use_tabnet_classifier = False
        if use_tabnet_classifier:
            n_size = 12
            self.clf = TabNetClassifier(optimizer_fn=torch.optim.AdamW,
                                        n_d=n_size, n_a=n_size, n_steps=5,
                                        optimizer_params=dict(lr=0.02),
                                        scheduler_params={"step_size": 10,
                                                          "gamma": 0.9},
                                        scheduler_fn=torch.optim.lr_scheduler.StepLR,
                                        )
        else:
            # Use LGBMClassifier:
            self.clf = LGBMClassifier(
                n_estimators=100,
                class_weight=class_weight,
                random_state=random_state,
                num_leaves=num_leaves,
                learning_rate=learning_rate,
                min_child_samples=min_child_samples,
                force_row_wise=True,
                verbose=-1,
            )

        # Use bagging instead of 'regular' classifier model:
        # self.base_clf = LGBMClassifier(class_weight=class_weight,
        #                                random_state=random_state,
        #                                num_leaves=num_leaves,
        #                                learning_rate=learning_rate,
        #                                min_child_samples=min_child_samples)
        # from sklearn.ensemble import BaggingClassifier
        # self.clf = BaggingClassifier(estimator=self.base_clf, n_estimators=10, random_state=random_state)

        # ============================================== Data and train ===============================================
        if self.processed_data:
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
            # if 'CURRENT_FIX_COMPONENT_IMAGE_NUMBER' in self.df.keys():
            #     input_data_points.append('CURRENT_FIX_COMPONENT_IMAGE_NUMBER')
            print_and_log('NOT USING CURRENT_FIX_COMPONENT_IMAGE_NUMBER')
        elif self.gen_data:
            input_data_points = [
                'TRIAL_FIX_INDEX',
                'EVENT_START',
                'EVENT_END',
                # 'EVENT',
                'FIXATION_DURATION',
                'SLICE_FIXATION_START',
                'SLICE_FIXATION_END',
                'SLICE_FIXATION_DURATION',
                # 'CURRENT_IMAGE',
                # 'EVENT_ZONE',
                'ZONE_X',
                'ZONE_Y',
                'CURRENT_FIX_PUPIL',
            ]
        else:
            input_data_points = [
                'CURRENT_FIX_INDEX',
                'Pupil_Size',
                'GAZE_IA_X',
                'GAZE_IA_Y',
                'SAMPLE_INDEX',
                'SAMPLE_START_TIME',
                'IN_BLINK',
                'IN_SACCADE',
            ]

        # For experiments training with added data columns:
        # input_data_points.append('RECORDING_SESSION_LABEL')
        # input_data_points.append('TRIAL_INDEX')

        if use_tabnet_classifier and self.test_df is None:
            # self.df['group'] = self.df[['RECORDING_SESSION_LABEL', 'TRIAL_INDEX', 'CURRENT_FIX_INDEX']].apply(
            #     lambda row: '_'.join(row.values.astype(str)), axis=1)
            self.df['group'] = self.df[['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']].apply(
                lambda row: '_'.join(row.values.astype(str)), axis=1)
            gss = GroupShuffleSplit(n_splits=1,
                                    test_size=test_size,
                                    random_state=random_state)
            split_indices = list(gss.split(X=self.df[input_data_points],
                                           y=self.df['target'],
                                           groups=self.df['group']))[0]
            train_index, test_index = split_indices
            x_train, x_test = self.df[input_data_points].iloc[train_index], self.df[input_data_points].iloc[test_index]
            y_train, y_test = self.df['target'].iloc[train_index], self.df['target'].iloc[test_index]
            self.df = pd.DataFrame([])
            self.df[input_data_points], self.df['target'] = x_train, y_train
            self.test_df = pd.DataFrame([])
            self.test_df[input_data_points], self.test_df['target'] = x_test, y_test

        # Would be used in cross-validation runs:
        acc_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        roc_auc_scores = []
        confusion_matrices = []
        y_preds = []
        y_tests = []

        # If confusion matrix is calculated, this parameter would be overridden:
        confusion_matrix_res = np.array([])

        if self.test_df is not None:
            print_and_log('Using self.test_df')
            x_train, y_train = self.df[input_data_points], self.df['target']
            x_test, y_test = self.test_df[input_data_points], self.test_df['target']

            if smote_type is not None:
                x_train, y_train = apply_smote_and_related(x=x_train, y=y_train, smote_type=smote_type)

            if use_tabnet_classifier:  # Could use the condition - self.clf.__class__.__name__ == 'TabNetClassifier'
                class_weight = {False: 1, True: (~self.df['target']).sum() / self.df['target'].sum()}
                loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([class_weight[False], class_weight[True]],
                                                                        dtype=torch.float32,
                                                                        device=torch.device('cuda:0')))
                # loss_fn = None

                x_train, x_test = x_train.to_numpy(), x_test.to_numpy()
                y_train, y_test = y_train.to_numpy(), y_test.to_numpy()
                # load_model = None
                # if load_model:
                #     self.clf.load_model('my_model.zip')

                unsupervised_model = TabNetPretrainer(
                    optimizer_fn=torch.optim.AdamW,
                    optimizer_params=dict(lr=2e-2),
                    mask_type='entmax'  # "sparsemax"
                )

                unsupervised_model.fit(
                    X_train=x_train,
                    eval_set=[x_test],
                    pretraining_ratio=0.8,
                )

                # noinspection PyArgumentList
                self.clf.fit(x_train, y_train, eval_set=[(x_test, y_test)], max_epochs=100, loss_fn=loss_fn,
                             from_unsupervised=unsupervised_model,
                             # batch_size=4096, virtual_batch_size=512)
                             batch_size=65536, virtual_batch_size=8192)
                # batch_size=262144, virtual_batch_size=32768)
                # batch_size=524288, virtual_batch_size=65536)
                now = datetime.now()
                datetime_string = now.strftime('D%y%m%dT%H%M')
                models_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')
                model_path = os.path.join(models_dir, f'tabnet_model_{datetime_string}')
                # noinspection PyUnresolvedReferences
                self.clf.save_model(path=model_path)
            else:
                self.clf.fit(x_train, y_train)
            y_pred = self.clf.predict(x_test)

            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
            precision = precision_score(y_true=y_test, y_pred=y_pred, average='binary')
            recall = recall_score(y_true=y_test, y_pred=y_pred, average='binary')
            f1 = f1_score(y_true=y_test, y_pred=y_pred, average='binary')
            roc_auc_score_res = roc_auc_score(y_true=y_test, y_score=y_pred)
        elif not cross_validate:
            # Old way, need to split randomly with groups of fixations.
            # x_train, x_test, y_train, y_test = train_test_split(self.df[input_data_points],
            #                                                     self.df['target'],
            #                                                     test_size=test_size,
            #                                                     random_state=random_state)

            # Group by the three keys and split by them - RECORDING_SESSION_LABEL, TRIAL_INDEX, CURRENT_FIX_INDEX:
            print_and_log('Splitting train / test data by RECORDING_SESSION_LABEL, TRIAL_INDEX, CURRENT_FIX_INDEX')
            #
            # self.df['group'] = self.df[['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']].apply(
            #     lambda row: '_'.join(row.values.astype(str)), axis=1)
            self.df['group'] = self.df[['RECORDING_SESSION_LABEL', 'TRIAL_INDEX', 'CURRENT_FIX_INDEX']].apply(
                lambda row: '_'.join(row.values.astype(str)), axis=1)
            gss = GroupShuffleSplit(n_splits=1,
                                    test_size=test_size,
                                    random_state=random_state)
            split_indices = list(gss.split(X=self.df[input_data_points],
                                           y=self.df['target'],
                                           groups=self.df['group']))[0]
            train_index, test_index = split_indices
            x_train, x_test = self.df[input_data_points].iloc[train_index], self.df[input_data_points].iloc[
                test_index]
            y_train, y_test = self.df['target'].iloc[train_index], self.df['target'].iloc[test_index]

            # x_test_file_path = 'data/temp/x_test.csv'
            # # x_test.to_csv(x_test_file_path, index=False)
            # x_test = pd.read_csv(x_test_file_path, nrows=None)
            # y_test_file_path = 'data/temp/y_test.csv'
            # # y_test.to_csv(y_test_file_path, index=False)
            # y_test = pd.read_csv(y_test_file_path, nrows=None)
            #
            # used_data = self.df[input_data_points]
            # used_data_with_indices = used_data.copy().reset_index()
            # merged_df = pd.merge(used_data_with_indices, x_test, how='outer', indicator=True)
            # x_train_with_index_column = merged_df.query('_merge == "left_only"').drop('_merge', axis=1)
            # original_indices = x_train_with_index_column['index'].tolist()
            # x_train = used_data[used_data.index.isin(original_indices)]
            # y_train = self.df['target'][self.df['target'].index.isin(original_indices)]

            if smote_type is not None:
                x_train, y_train = apply_smote_and_related(x=x_train, y=y_train, smote_type=smote_type)

            self.clf.fit(x_train, y_train)
            y_pred = self.clf.predict(x_test)

            acc = accuracy_score(y_true=y_test, y_pred=y_pred)
            precision = precision_score(y_true=y_test, y_pred=y_pred, average='binary')
            recall = recall_score(y_true=y_test, y_pred=y_pred, average='binary')
            f1 = f1_score(y_true=y_test, y_pred=y_pred, average='binary')
            roc_auc_score_res = roc_auc_score(y_true=y_test, y_score=y_pred)
        else:
            print_and_log(f'Running cross-validation with n_splits = {cross_validation_n_splits}')
            if split_by_participants:
                print_and_log('Splitting train / test data by participants')
                split_indices = []
                unique_participants = self.df['RECORDING_SESSION_LABEL'].unique()
                # split_test_size = int(len(unique_participants) * test_size)  # If I want to use the function test_size
                split_test_size = int(len(unique_participants) * 0.2)  # If I want to use a fixed test size (20%)
                print_and_log(
                    f'Using split_test_size = {split_test_size} out of {len(unique_participants)} participants\n'
                    f'In percentage, this is {split_test_size / len(unique_participants) * 100:.2f}%')
                np.random.seed(seed=random_state)

                test_participants_labels = stratified_group_split_independent_unique(
                    df=self.df,
                    target_col='target',
                    group_col='RECORDING_SESSION_LABEL',
                    test_size=split_test_size,
                    n_splits=cross_validation_n_splits
                )
                for test_participants_label in test_participants_labels:
                    train_indices = self.df[~self.df['RECORDING_SESSION_LABEL'].isin(test_participants_label)].index
                    test_indices = self.df[self.df['RECORDING_SESSION_LABEL'].isin(test_participants_label)].index
                    split_indices.append((train_indices, test_indices))
            else:
                if participants_to_train_on is not None:
                    print_and_log(f'Using given participants list: {participants_to_train_on}')
                    unique_participants = participants_to_train_on
                else:
                    unique_participants = self.df['RECORDING_SESSION_LABEL'].unique()

                if predict_all_train_data:
                    # Create five unique, non-overlapping splits, test it on them, saving preds to df then to file:
                    splits_dict = {ind: [] for ind in range(5)}
                    test_participants = unique_participants
                    if participants_to_test_on is not None:
                        test_participants = participants_to_test_on
                    for participant in test_participants:
                        participant_indices = self.df[self.df['RECORDING_SESSION_LABEL'] == participant].index
                        shuffled_indices = np.random.permutation(participant_indices)
                        split_indices = np.array_split(shuffled_indices, 5)
                        for ind in range(5):
                            splits_dict[ind].extend(split_indices[ind])
                    for ind in range(5):
                        np.random.shuffle(splits_dict[ind])
                    split_indices = [(list(set(self.df.index) - set(splits_dict[ind])), splits_dict[ind])
                                     for ind in range(5)]
                else:
                    # Randomly split the data comprising 80% of the data for each participant,
                    # rather than just 80% of the total data:
                    print_and_log('Splitting train / test data by 80% of each participant\'s data')
                    if not self.gen_data:
                        # self.df['group'] = self.df[['RECORDING_SESSION_LABEL', 'TRIAL_INDEX',
                        #                             'CURRENT_FIX_INDEX']].apply(
                        #     lambda row: '_'.join(row.values.astype(str)), axis=1)
                        self.df['group'] = self.df[['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']].apply(
                            lambda row: '_'.join(row.values.astype(str)), axis=1)
                    if completely_random_data_split:
                        # gss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                        gss = GroupShuffleSplit(n_splits=1, test_size=test_size)
                    else:
                        # gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                        gss = GroupShuffleSplit(n_splits=1, test_size=test_size)
                    split_indices = []
                    for _ in range(cross_validation_n_splits):
                        train_indices = []
                        test_indices = []
                        for participant in unique_participants:
                            participant_indices = self.df[self.df['RECORDING_SESSION_LABEL'] == participant].index
                            if completely_random_data_split:
                                participant_split_indices = gss.split(X=participant_indices)
                            else:
                                participant_split_indices = gss.split(X=participant_indices,
                                                                      groups=self.df['group'].iloc[participant_indices])
                            participant_split_indices = list(participant_split_indices)
                            participant_train_indices, participant_test_indices = participant_split_indices[0]
                            train_indices.extend(participant_indices[participant_train_indices])
                            test_indices.extend(participant_indices[participant_test_indices])
                        if participants_to_test_on is not None:
                            test_indices = self.df['RECORDING_SESSION_LABEL'].isin(participants_to_test_on)
                            participant_test_indices = self.df[test_indices].index
                            test_indices = list(participant_test_indices)
                            # Overwrite the test indices
                        np.random.shuffle(train_indices)
                        np.random.shuffle(test_indices)
                        split_indices.append((train_indices, test_indices))

            # all_probabilities = []
            now = datetime.now()
            datetime_string = now.strftime('D%y%m%dT%H%M')
            for split_ind, (train_index, test_index) in tqdm(iterable=enumerate(split_indices),
                                                             desc='Iterating over different data splits',
                                                             total=len(split_indices)):
                x_train, x_test = (self.df[input_data_points].iloc[train_index],
                                   self.df[input_data_points].iloc[test_index])
                y_train, y_test = (self.df['target'].iloc[train_index],
                                   self.df['target'].iloc[test_index])

                if participants_to_test_on is not None:  # For testing on given participants:
                    indices = self.df[self.df['RECORDING_SESSION_LABEL'].isin(participants_to_test_on)].index
                    common_indices = indices.intersection(x_test.index)
                    filtered_x_test = x_test.loc[common_indices]
                    x_test = filtered_x_test
                    filtered_y_test = y_test[common_indices]
                    y_test = filtered_y_test

                if smote_type is not None:
                    x_train, y_train = apply_smote_and_related(x=x_train, y=y_train, smote_type=smote_type)

                self.clf.fit(x_train, y_train)
                y_pred = self.clf.predict(x_test)

                # Group results by RECORDING_SESSION_LABEL:
                if group_by_participants := False:
                    print_and_log('Grouping results by \'RECORDING_SESSION_LABEL\'')
                    results_df = pd.DataFrame({
                        'prediction': y_pred,
                        'RECORDING_SESSION_LABEL': self.df['RECORDING_SESSION_LABEL'].iloc[test_index],
                        'target': self.df['target'].iloc[test_index],
                    })

                    def threshold_and_compare(group, threshold=0.5):
                        mean_prediction = group['prediction'].mean()
                        threshold_prediction = mean_prediction > threshold
                        target = group['target'].iloc[0]  # Assuming all targets in a group are the same
                        return pd.Series({
                            'predicted': threshold_prediction,
                            'target': target,
                            'correct': threshold_prediction == target
                        })

                    grouped_results = results_df.groupby('RECORDING_SESSION_LABEL').apply(threshold_and_compare)
                    y_pred = grouped_results['predicted'].to_numpy()
                    y_test = grouped_results['target'].to_numpy()

                if False and predict_all_train_data:
                    # Save predictions back to the df:
                    self.df.loc[test_index, 'predicted'] = y_pred

                if group_by_fix_comp_image_file := False:
                    print_and_log('Grouping nodules by \'CURRENT_FIX_COMPONENT_IMAGE_FILE\'')
                    # Consider rows where SLICE_TYPE==NODULE_SLICE having the same 'CURRENT_FIX_COMPONENT_IMAGE_FILE' as
                    # one data rows and benchmark on them accordingly:
                    df_test = self.df.iloc[test_index].reset_index(drop=True)
                    test_nodules = df_test[df_test['SLICE_TYPE'] == 'NODULE_SLICE']

                    def apply_threshold(indices, threshold=0.5):
                        # Convert lists of boolean to int, then calculate mean to see if it exceeds a threshold
                        test_result = np.mean(y_test[indices]) >= threshold
                        pred_result = np.mean(y_pred[indices]) >= threshold
                        return test_result, pred_result

                    grouped_test_nodules_results = test_nodules.groupby('CURRENT_FIX_COMPONENT_IMAGE_FILE').apply(
                        lambda x: apply_threshold(x.index)).reset_index(drop=True)

                    new_y_test = [result[0] for result in grouped_test_nodules_results]
                    new_y_pred = [result[1] for result in grouped_test_nodules_results]

                    # Handle non-NODULE_SLICE:
                    test_non_nodules = df_test[df_test['SLICE_TYPE'] != 'NODULE_SLICE']
                    if group_non_nodules := True:
                        # Group by 'CURRENT_FIX_COMPONENT_IMAGE_FILE':
                        grouped_test_non_nodules_results = test_non_nodules.groupby(
                            'CURRENT_FIX_COMPONENT_IMAGE_FILE').apply(lambda x: apply_threshold(x.index)).reset_index(
                            drop=True)
                        new_y_test.extend([result[0] for result in grouped_test_non_nodules_results])
                        new_y_pred.extend([result[1] for result in grouped_test_non_nodules_results])
                    else:
                        # Append rows directly to the new_y_test and new_y_pred lists:
                        new_y_test.extend(y_test[test_non_nodules.index])
                        new_y_pred.extend(y_pred[test_non_nodules.index])

                    y_test = np.array(new_y_test)
                    y_pred = np.array(new_y_pred)

                # all_probabilities.extend(probabilities)
                # probabilities = self.clf.predict_proba(x_test)[:, 1]
                # roc_auc_scores.append(roc_auc_score(y_test, probabilities))
                roc_auc_scores.append(-1)
                acc_scores.append(accuracy_score(y_test, y_pred))
                precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1])
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

                conf_matrix = confusion_matrix(y_test, y_pred)
                if conf_matrix.shape == (2, 2):
                    confusion_matrices.append(conf_matrix)
                y_preds.append(y_pred)
                y_tests.append(y_test)

                if save_models or save_predicted_as_true_data_rows:
                    if completely_random_data_split:
                        data_split_name = 'with_completely_random_data_split'
                    else:
                        data_split_name = 'with_REC_TRIAL_FIX_group_shuffle_split'
                    parent_folder_name = f'lgbm_model_approach_{self.approach_num}_{data_split_name}_{datetime_string}'
                    model_path = Path('models') / parent_folder_name / f'split_{split_ind}.ckpt'
                    model_path.parent.mkdir(parents=True, exist_ok=True)

                    if save_models:  # Save the model for later use
                        self.clf.booster_.save_model(str(model_path))

                    if save_predicted_as_true_data_rows:
                        predicted_rows = self.df.iloc[test_index][y_pred]
                        # Set 'original_index' column as the index of the dataframe:
                        predicted_rows = predicted_rows.set_index('original_index')
                        predicted_rows = predicted_rows.sort_index()
                        pred_rows_csv_path = model_path.parent / f'predicted_rows_for_slice_{split_ind}.csv'
                        predicted_rows.to_csv(str(pred_rows_csv_path), index=True)

            if False and predict_all_train_data:
                # Save df to file with all the predictions:
                now = datetime.now()
                datetime_string = now.strftime('D%y%m%dT%H%M')
                df_csv_dir = Path('data') / 'data_with_preds'
                file_name = Path(self.data_file_path).stem

                # Save the processed file with predictions to a csv file:
                # processed_df_csv_path = df_csv_dir / f'Processed_file_{file_name}_with_preds_{datetime_string}.csv'
                # self.df.to_csv(str(processed_df_csv_path), index=False)

                # Load the original file:
                original_df = pd.read_csv(self.data_file_path)
                # Add the predictions to the original file:
                majority_predictions = self.df.groupby('original_index')['predicted'].agg(lambda x: x.mode()[0])
                original_df['predicted'] = original_df.index.map(majority_predictions)
                # Save the original file with the predictions:
                original_df_csv_path = df_csv_dir / f'Original_file_{file_name}_with_preds_{datetime_string}.csv'
                original_df.to_csv(str(original_df_csv_path), index=False)
                print_and_log(f'Saved the original file with predictions to {original_df_csv_path}')

            confusion_matrix_res = np.mean(confusion_matrices, axis=0).round().astype(int)
            y_pred = np.concatenate(y_preds)
            y_test = np.concatenate(y_tests)

            # This values would be returned from this training function:
            acc = np.mean(acc_scores)
            precision = np.mean(precision_scores, axis=0)
            recall = np.mean(recall_scores, axis=0)
            # print('=' * 50)
            # print('F1 scores for all splits:')
            # for ind, score in enumerate(f1_scores):
            #     print(f'Split {ind}: {score}')
            # print('\n', '=' * 50)
            f1 = np.mean(f1_scores, axis=0)
            # roc_auc_score_res = np.mean(roc_auc_scores)
            roc_auc_score_res = [-1]

        if plot_confusion_matrix:
            def plot_confusion_matrix(data, annot, fmt, title):
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                sns.heatmap(data, annot=annot, fmt=fmt, cmap='Blues', cbar=False,
                            xticklabels=['Predicted 0', 'Predicted 1'],
                            yticklabels=['Actual 0', 'Actual 1'])
                plt.title(title)
                plt.show()

            if not cross_validate:
                confusion_matrix_res = confusion_matrix(y_true=y_test, y_pred=y_pred)

            if plot_with_percentage := True:
                confusion_matrix_percentage = confusion_matrix_res / np.sum(confusion_matrix_res) * 100
                annot = np.array([["{0:.2f}%".format(value) for value in row]
                                  for row in confusion_matrix_percentage])
                title = (f'Confusion Matrix (%). {self.clf.__class__.__name__} '
                         f'model with class_weight={self.clf.class_weight}')
                plot_confusion_matrix(confusion_matrix_percentage, annot, '', title)
                if save_plot:
                    self.save_plot_func(title=title)
            if plot_with_abs_values := True:
                plt.clf()  # Clear figure before plotting new matrix
                title = f'Confusion Matrix. {self.clf.__class__.__name__} model with class_weight={self.clf.class_weight}'
                plot_confusion_matrix(confusion_matrix_res, True, 'd', title)
                if save_plot:
                    self.save_plot_func(title=title)

        if print_stats:

            print_and_log(f'The full classification report is:\n{classification_report(y_true=y_test, y_pred=y_pred, digits=5)}')
        acc = np.mean(acc_scores)
        precision = np.mean(precision_scores, axis=0)
        recall = np.mean(recall_scores, axis=0)
        f1 = np.mean(f1_scores, axis=0)
        roc_auc_score_res = np.mean(roc_auc_scores)
        cm = confusion_matrix(y_test, y_pred)

        print("Confusion Matrix:")
        print("   Predicted 0  Predicted 1")
        print(f"Actual 0   {cm[0, 0]:<10} {cm[0, 1]:<10}")
        print(f"Actual 1   {cm[1, 0]:<10} {cm[1, 1]:<10}")
        return acc, precision, recall, f1, roc_auc_score_res
# label 2
# The full classification report is:
#               precision    recall  f1-score   support
#
#        False    0.98782   0.99808   0.99293      9915
#         True    0.00000   0.00000   0.00000       122
#
#     accuracy                        0.98595     10037
#    macro avg    0.49391   0.49904   0.49646     10037
# weighted avg    0.97581   0.98595   0.98086     10037
# Confusion Matrix:
#    Predicted 0  Predicted 1
# Actual 0   9896       19
# Actual 1   122        0

# label 3
# The full classification report is:
#               precision    recall  f1-score   support
#
#        False    0.97227   0.99856   0.98524     14609
#         True    0.08696   0.00478   0.00907       418
#
#     accuracy                        0.97092     15027
#    macro avg    0.52962   0.50167   0.49716     15027
# weighted avg    0.94765   0.97092   0.95809     15027
#
# Confusion Matrix:
#    Predicted 0  Predicted 1
# Actual 0   14588      21
# Actual 1   416        2


if __name__ == '__main__':

    # If used, will predict all the input train data and save it to a csv file:
    parser = argparse.ArgumentParser(description='Train a model on the given data, predict it and save to a csv file.')
    parser.add_argument('--data_file_path', type=str,
                        help='The path to the data file to train the model on.')
    args = parser.parse_args()
    data_file_path = args.data_file_path

    if data_file_path is not None:
        used_data_file_path = data_file_path
        predict_all_train_data = True
        print_stats = False
        approach_num = 8
    else:
        nodule_categorized_rad_s1_s18_file_path = 'data/Categorized_Fixation_Data_1_18.csv'
        used_data_file_path = nodule_categorized_rad_s1_s18_file_path

        # ekg_001_file_path = 'data/ECG/ML_ECGData_P001.csv'
        # used_data_file_path = ekg_001_file_path

        # six_participants = 'data/OLD_Nodule_Categorized_Fixation_Data_7_12.xlsx'
        # used_data_file_path = six_participants

        approach_num = 6
        # generalist_expert_med_student_file_path = 'data/group_differentiation/Generalists_Experts_MedStudents_Combined.csv'
        # approach_num = 11
        # used_data_file_path = generalist_expert_med_student_file_path
        predict_all_train_data = False
        print_stats = True

    print_and_log(f'Running with approach {approach_num} and data file path {used_data_file_path}')

    init_kwargs = dict(
        data_file_path=used_data_file_path,
        test_data_file_path=None,
        augment=False,
        join_train_and_test_data=False,
        normalize=True,
        remove_surrounding_to_hits=0,
        update_surrounding_to_hits=0,
        approach_num=approach_num,
    )
    ident_sub_rec = IdentSubRec(**init_kwargs)

    # Baseline parameters:
    train_kwargs = dict(test_size=0.2, num_leaves=300, learning_rate=0.1, min_child_samples=44, print_stats=print_stats,
                        plot_confusion_matrix=False, save_plot=False, cross_validate=False, cross_validation_n_splits=10,
                        split_by_participants=False, completely_random_data_split=False, smote_type=SmoteType.none,
                        save_models=False, save_predicted_as_true_data_rows=False,
                        predict_all_train_data=predict_all_train_data)
    ident_sub_rec.train(**train_kwargs)

    # Regular train:
    # labels = list(ident_sub_rec.df['RECORDING_SESSION_LABEL'].unique())
    # for label in labels:
    #     ident_sub_rec = IdentSubRec(**init_kwargs)
    #     print(f"############################LABEL {label}#########################")
    #     ident_sub_rec.df = ident_sub_rec.df[ident_sub_rec.df['RECORDING_SESSION_LABEL'] == label]
    #     ident_sub_rec.train(**train_kwargs)
    #
