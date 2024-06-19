import numpy as np
from tqdm import tqdm

from tfd_utils.visualization_utils import plot_model_stats_per_x_results


def model_stats_per_training_size(ident_sub_rec, one_plot: bool = True, save_plot: bool = True):
    test_sizes = np.arange(0.05, 1, 0.05)
    results = [ident_sub_rec.train(test_size=test_size,
                                   print_stats=False,
                                   plot_confusion_matrix=False,
                                   save_plot=False) for test_size in tqdm(test_sizes)]
    plot_model_stats_per_x_results(x_axis_vals=test_sizes,
                                   results=results,
                                   title=f'Stats of {ident_sub_rec.clf.__class__.__name__} model with '
                                         f'class_weight={ident_sub_rec.clf.class_weight} on different test sizes '
                                         f'(train complement to 1)',
                                   xlabel='test sizes',
                                   one_plot=one_plot,
                                   save_plot=save_plot)


def model_stats_per_weight_classes(ident_sub_rec, one_plot: bool = True, save_plot: bool = True):
    # true_label_weights = np.arange(1, 51, 1)
    true_label_weights = np.arange(51, 101, 1)
    # true_label_weights = np.arange(1, 101, 1)
    results = [ident_sub_rec.train(test_size=0.2,
                                   class_weight={False: 1, True: true_label_weight},
                                   print_stats=False,
                                   plot_confusion_matrix=False,
                                   save_plot=False) for true_label_weight in tqdm(true_label_weights)]
    plot_model_stats_per_x_results(x_axis_vals=true_label_weights,
                                   results=results,
                                   title=f'Stats of {ident_sub_rec.clf.__class__.__name__} model on different true '
                                         f'labels class weights from {true_label_weights.min()} '
                                         f'to {true_label_weights.max()} (false labels class weight is always 1)',
                                   xlabel='true labels class weights',
                                   one_plot=one_plot,
                                   save_plot=save_plot)


def model_stats_per_learning_rate(ident_sub_rec, one_plot: bool = True, save_plot: bool = True,
                                  learning_rates=np.arange(0.01, 0.11, 0.01)):
    # learning_rates = np.arange(0.01, 0.21, 0.01)
    # learning_rates = np.arange(0.21, 0.41, 0.01)
    results = [ident_sub_rec.train(test_size=0.2,
                                   num_leaves=300,
                                   learning_rate=learning_rate,
                                   print_stats=False,
                                   plot_confusion_matrix=False,
                                   save_plot=False,
                                   cross_validate=True,
                                   cross_validation_n_splits=10,
                                   split_by_participants=False,
                                   use_smote=False) for learning_rate in tqdm(learning_rates)]

    plot_model_stats_per_x_results(x_axis_vals=learning_rates,
                                   results=results,
                                   title=f'Stats of {ident_sub_rec.clf.__class__.__name__} model with '
                                         f'class_weight={ident_sub_rec.clf.class_weight} on different learning rates '
                                         f'from {learning_rates.min():.2} to {learning_rates.max():.2}',
                                   xlabel='learning rates',
                                   one_plot=one_plot,
                                   save_plot=save_plot)


def model_stats_per_num_leaves(ident_sub_rec, one_plot: bool = True, save_plot: bool = True,
                               num_leaves_list=np.arange(101, 151, 1)):
    # num_leaves_list = np.arange(10, 51, 1)
    # num_leaves_list = np.arange(51, 101, 1)
    results = [ident_sub_rec.train(test_size=0.2,
                                   num_leaves=num_leaves,
                                   learning_rate=0.1,
                                   min_child_samples=20,
                                   print_stats=False,
                                   plot_confusion_matrix=False,
                                   save_plot=False,
                                   cross_validate=True,
                                   cross_validation_n_splits=10) for num_leaves in tqdm(num_leaves_list)]
    plot_model_stats_per_x_results(x_axis_vals=num_leaves_list,
                                   results=results,
                                   title=f'Stats of {ident_sub_rec.clf.__class__.__name__} model with '
                                         f'class_weight={ident_sub_rec.clf.class_weight} on different number of leaves '
                                         f'from {num_leaves_list.min()} to {num_leaves_list.max()}',
                                   xlabel='num leaves',
                                   one_plot=one_plot,
                                   save_plot=save_plot)


def model_stats_per_min_child_samples(ident_sub_rec, one_plot: bool = True, save_plot: bool = True):
    min_child_samples_list = np.arange(1, 51, 1)

    results = [ident_sub_rec.train(test_size=0.2,
                                   num_leaves=300,
                                   learning_rate=0.1,
                                   min_child_samples=min_child_samples,
                                   print_stats=False,
                                   plot_confusion_matrix=False,
                                   save_plot=False,
                                   cross_validate=True,
                                   cross_validation_n_splits=10,
                                   split_by_participants=False) for min_child_samples in tqdm(min_child_samples_list)]
    plot_model_stats_per_x_results(x_axis_vals=min_child_samples_list,
                                   results=results,
                                   title=f'Stats of {ident_sub_rec.clf.__class__.__name__} model with '
                                         f'class_weight={ident_sub_rec.clf.class_weight} on different minimum number '
                                         f'of child samples',
                                   xlabel='min child samples',
                                   one_plot=one_plot,
                                   save_plot=save_plot)
