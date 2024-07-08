import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from lightgbm import LGBMClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report,precision_recall_curve
# from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from tqdm import tqdm

from tfd_utils.data_utils import get_df_for_training
from tfd_utils.logger_utils import print_and_log
from tfd_utils.model_stats_utils import model_stats_per_training_size, model_stats_per_weight_classes, \
    model_stats_per_learning_rate, model_stats_per_num_leaves, model_stats_per_min_child_samples
from tfd_utils.visualization_utils import save_plot_func, visualize_data, visualize_model, compare_old_to_new_data


# from tfd_utils.visualization_utils import visualize_data, save_plot_func, visualize_model, compare_old_to_new_data


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

        self.df, self.processed_data = self.get_df_for_training(
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
        # self.df = self.df[self.df['RECORDING_SESSION_LABEL'] == 1]

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

    def train(self,
              test_size: float = 0.2,
              class_weight: Optional[dict] = None,
              num_leaves: int = 31,
              learning_rate: float = 0.1,
              min_child_samples: int = 20,
              print_stats: bool = True,
              plot_confusion_matrix: bool = True,
              save_plot: bool = False,
              random_state: int = 666,
              cross_validate: bool = True,
              cross_validation_n_splits: int = 10,
              split_by_participants: bool = False,
              completely_random_data_split: bool = False,
              smote_type: Optional[SmoteType] = False,
              save_models: bool = False,
              save_predicted_as_true_data_rows: bool = False):
        print_and_log(f'Running with num_leaves={num_leaves}, learning_rate={learning_rate}, '
                      f'min_child_samples={min_child_samples}, random_state={random_state}, '
                      f'cross_validate={cross_validate}, cross_validation_n_splits={cross_validation_n_splits}')
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



        #need GPU
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
            if 'CURRENT_FIX_COMPONENT_IMAGE_NUMBER' in self.df.keys():
                input_data_points.append('CURRENT_FIX_COMPONENT_IMAGE_NUMBER')
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


        if use_tabnet_classifier and self.test_df is None:
            self.df['group'] = self.df[['RECORDING_SESSION_LABEL', 'TRIAL_INDEX', 'CURRENT_FIX_INDEX']].apply(
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
            pass
        else:
            print_and_log(f'Running cross-validation with n_splits = {cross_validation_n_splits}')
            if split_by_participants:
                print_and_log('Splitting train / test data by participants')
                split_indices = []
                unique_participants = self.df['RECORDING_SESSION_LABEL'].unique()
                # split_test_size = int(len(unique_participants) * test_size)  # If I want to use the function test_size
                split_test_size = 2
                print_and_log(
                    f'Using split_test_size = {split_test_size} out of {len(unique_participants)} participants\n'
                    f'In percentage, this is {split_test_size / len(unique_participants) * 100:.2f}%')
                np.random.seed(seed=random_state)
                test_participants_labels = np.random.choice(unique_participants,
                                                            size=(cross_validation_n_splits, split_test_size))
                for test_participants_label in test_participants_labels:
                    train_indices = self.df[~self.df['RECORDING_SESSION_LABEL'].isin(test_participants_label)].index
                    test_indices = self.df[self.df['RECORDING_SESSION_LABEL'].isin(test_participants_label)].index
                    split_indices.append((train_indices, test_indices))
            else:

                print_and_log('Splitting train / test data by 80% of each participants data')
                unique_participants = self.df['RECORDING_SESSION_LABEL'].unique()
                self.df['group'] = self.df[['RECORDING_SESSION_LABEL', 'TRIAL_INDEX', 'CURRENT_FIX_INDEX']].apply(
                    lambda row: '_'.join(row.values.astype(str)), axis=1)
                if completely_random_data_split:
                    gss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                else:
                    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
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

                if smote_type is not None:
                    x_train, y_train = apply_smote_and_related(x=x_train, y=y_train, smote_type=smote_type)

                self.clf.fit(x_train, y_train)
                y_pred = self.clf.predict(x_test)
                probabilities = self.clf.predict_proba(x_test)[:, 1]
                # all_probabilities.extend(probabilities)

                acc_scores.append(accuracy_score(y_test, y_pred))
                precision_scores.append(precision_score(y_test, y_pred, average='binary'))
                recall_scores.append(recall_score(y_test, y_pred, average='binary'))
                f1_scores.append(f1_score(y_test, y_pred, average='binary'))
                roc_auc_scores.append(roc_auc_score(y_test, probabilities))
                confusion_matrices.append(confusion_matrix(y_test, y_pred))
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



            confusion_matrix_res = np.mean(confusion_matrices, axis=0).round().astype(int)
            y_pred = np.concatenate(y_preds)
            y_test = np.concatenate(y_tests)

            # This values would be returned from this training function:
            acc = np.mean(acc_scores)
            precision = np.mean(precision_scores)
            recall = np.mean(recall_scores)
            f1 = np.mean(f1_scores)
            roc_auc_score_res = np.mean(roc_auc_scores)

        if plot_confusion_matrix:
            if not cross_validate:
                confusion_matrix_res = confusion_matrix(y_true=y_test, y_pred=y_pred)
            title = f'Confusion Matrix. {self.clf.__class__.__name__} model with class_weight={self.clf.class_weight}'
            plt.title(title)
            sns.heatmap(confusion_matrix_res, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=['Predicted 0', 'Predicted 1'],
                        yticklabels=['Actual 0', 'Actual 1'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            if save_plot:
                self.save_plot_func(title=title)
            plt.show()

        if print_stats:
            if self.clf.__class__.__name__ not in ('NeuralNetBinaryClassifier', 'TabNetClassifier'):
                set_class_weight = self.base_clf.class_weight if self.base_clf is not None else self.clf.class_weight
                print_and_log(f'Using {self.clf.__class__.__name__} model with class_weight={set_class_weight}')
            if self.test_df is not None or not cross_validate:
                print_and_log(f'Results are:\n'
                              f'Accuracy {acc * 100:.2f}%\n'
                              f'Precision {precision * 100:.2f}%\n'
                              f'Recall {recall * 100:.2f}%\n'
                              f'F1 score {f1 * 100:.2f}%\n'
                              f'ROC AUC score {roc_auc_score_res * 100:.2f}%\n\n'
                              f'Used parameters are:\n'
                              f'test_size={test_size}\n'
                              f'data points keys = {input_data_points}')
            else:
                print_and_log(f'Used parameters are:\n'
                              f'test_size={test_size}\n'
                              f'data points keys = {input_data_points}')
                for operation in (np.mean, np.max, np.min, np.std):
                    print_and_log(f'{operation.__name__} results are:')
                    acc = operation(acc_scores)
                    precision = operation(precision_scores)
                    recall = operation(recall_scores)
                    f1 = operation(f1_scores)
                    roc_auc_score_res = operation(roc_auc_scores)
                    print_and_log(f'Accuracy {acc * 100:.2f}%\n'
                                  f'Precision {precision * 100:.2f}%\n'
                                  f'Recall {recall * 100:.2f}%\n'
                                  f'F1 score {f1 * 100:.2f}%\n'
                                  f'ROC AUC score {roc_auc_score_res * 100:.2f}%\n\n')
            print_and_log(f'The full classification report is:\n{classification_report(y_true=y_test, y_pred=y_pred)}')
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
        approach_num=7,
    )

    ident_sub_rec = IdentSubRec(**categorized_rad_init_kwargs)
    train_kwargs = dict(test_size=0.2, num_leaves=300, learning_rate=0.1, min_child_samples=44, print_stats=True,
                        plot_confusion_matrix=False, save_plot=False, cross_validate=True, cross_validation_n_splits=10,
                        split_by_participants=False, completely_random_data_split=False, smote_type=SmoteType.none,
                        save_models=False, save_predicted_as_true_data_rows=True)
    ident_sub_rec.train(**train_kwargs)