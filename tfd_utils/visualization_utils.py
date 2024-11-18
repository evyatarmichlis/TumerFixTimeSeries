import os
from typing import Optional, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree
from tabulate import tabulate
from tqdm import tqdm

from tfd_utils.logger_utils import print_and_log


def save_plot_func(title: str, data: Optional = None):
    """
    Save plot and if data is provided, save it as npy file.
    :param title: How the plot will be saved, processed a bit before used as a file path
    :param data: The data used for plotting
    :return: None
    """
    if not title.endswith('.png'):
        title = f'{title}.png'
    title = title.replace(' ', '_')
    fig_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots', title)
    fig = plt.gcf()
    fig.set_size_inches(27.48, 15.43)
    fig.savefig(fig_file_path, bbox_inches='tight')

    if data is not None:
        np.save(file=f'{fig_file_path[:-len(".png")]}.npy', arr=data)


def visualize_data(df: pd.DataFrame,
                   data_file_path: Union[str, List[str]],
                   start_grid: int = 1,
                   end_grid: int = 12,
                   save_plot: bool = False,
                   show_plot: bool = True):
    on_target = (df['target'] == 1).sum()
    not_on_target = (df['target'] == 0).sum()
    print_and_log(f'{on_target} data rows are where the position of the cell is on the target to recognize')
    print_and_log(f'{not_on_target} data rows are where the position of the cell is NOT on the target to recognize')

    # num_of_cells = (end_grid - start_grid + 1) ** 2

    def map_point_keys_to_clean_named_df(x_key, y_key):
        return df[[x_key, y_key]].rename(columns={x_key: 'x', y_key: 'y'})

    if all(key in df.keys() for key in ('Zones_X', 'Zones_Y')):
        x_key, y_key = 'Zones_X', 'Zones_Y'
    else:
        x_key, y_key = 'GAZE_IA_X', 'GAZE_IA_Y'
    df_all_points = map_point_keys_to_clean_named_df(x_key=x_key, y_key=y_key)

    # Take only valid points:
    df_all_points = df_all_points[(df_all_points['x'] > 0) & (df_all_points['y'] > 0)]

    # def print_unique_points_stats(df_for_prints, cells_num: int = 12 ** 2):
    #     df_for_prints = df_for_prints.copy()  # Changes here shouldn't affect the given DataFrame
    #     # Apply conversion of int values to two digit strings:
    #     for key in df_for_prints.keys():
    #         df_for_prints[key] = df_for_prints[key].apply('{:02d}'.format)
    #     # Get uniques:
    #     unique_cells = df_for_prints.agg('-'.join, axis=1).unique()
    #     # Order, I've converted into 2 digit strings for the ordering:
    #     unique_cells = sorted(unique_cells)
    #
    #     print_and_log(f'There are {len(df_for_prints)} unique cell values in the whole data')
    #     print_and_log(f'Grid cells which have any value:\n{unique_cells}')
    #     print_and_log(f'There are {len(unique_cells)} / {cells_num} cells with values in the data '
    #                   f'({len(unique_cells) / cells_num * 100:.2f}%). The others - not even 1 value in the data')
    #
    # print_unique_points_stats(df_for_prints=df_all_points, cells_num=num_of_cells)

    # Plotting a 2D histogram:
    if isinstance(data_file_path, list):
        data_file_path = '/'.join(data_file_path)
    label = (f'Cells position occurrences in {os.path.basename(data_file_path)}.'
             f' There are {len(df_all_points)} unique cell values')
    plt.title(label=label)
    bins = (range(start_grid, end_grid + 2), range(start_grid, end_grid + 2))
    # Create histogram and both x and y centers just to plot cell counts:
    hist, xedges, yedges = np.histogram2d(df_all_points['x'], df_all_points['y'], bins=bins)
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2

    sns.histplot(data=df_all_points, x='x', y='y',
                 bins=bins,
                 cbar=True).invert_yaxis()

    # Add the cell count to the sns 2D histogram plot:
    for i in range(len(xcenters)):
        for j in range(len(ycenters)):
            plt.text(xcenters[i], ycenters[j], int(hist[i, j]), ha='center', va='center', color='white')

    plt.grid()
    plt.xticks(range(start_grid, end_grid + 2))
    plt.yticks(range(start_grid, end_grid + 2))
    if save_plot:
        save_plot_func(title=label.replace('\n', ' '))
    if show_plot:
        plt.show()


def visualize_model(clf):
    if clf is None:
        raise ValueError('Provided classifier model is None')

    plot_tree(decision_tree=clf.estimators_[0], filled=True, max_depth=1)
    plt.show()


def compare_old_to_new_data(old_df, df):
    if old_df is None or df is None:
        raise ValueError('Please provide a Non-None old_df and df parameters')

    if len(df) != len(old_df):
        print(f'df have length of {len(df)} but old_df have length of {len(old_df)}')

    compare_up_to = min(len(df), len(old_df))

    df1 = df[:compare_up_to]
    df2 = old_df[:compare_up_to]

    if df1.columns.tolist() != df2.columns.tolist():
        print_and_log(f'Headers differ on the 2 data frames:\n{df1.columns.tolist()}\nVS\n{df2.columns.tolist()}')
        return

    col_diffs = []
    for column in tqdm(df1.columns):
        # noinspection PyTypeChecker
        col_diffs_df: pd.DataFrame = (df2[column] != df1[column])
        col_diff = 100 * col_diffs_df.sum() / df1[column].size
        col_diffs.append(f'{col_diff:.2f}%')

    table = list(zip(df1.columns, col_diffs))
    print_and_log(tabulate(table, headers=['Columns', 'Diffs'], tablefmt='orgtbl'))


def plot_model_stats_per_x_results(x_axis_vals,
                                   results,
                                   title: str,
                                   xlabel: str,
                                   font_size: int = 20,
                                   one_plot: bool = True,
                                   save_plot: bool = False):
    results_names = ['acc', 'precision', 'recall', 'f1']  # This is the first four results_old from self.train by order
    ncols = 1  # To calm down PyCharm that I won't use this parameter without initialization.
    if one_plot:
        fig, axs = plt.subplots()
    else:
        nrows, ncols = 2, 2
        assert nrows * ncols >= len(results_names), 'Must plot at least as much subplots as the used results_old'
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    fig.suptitle(title)
    for res_ind, res_name in enumerate(results_names):
        rel_res = [res[res_ind] for res in results]
        if one_plot:
            rel_axis = axs
            rel_axis.plot(x_axis_vals, rel_res, label=res_name)
        else:
            rel_axis = axs[res_ind // ncols, res_ind % ncols]
            rel_axis.plot(x_axis_vals, rel_res)
            rel_axis.set_title(res_name)
        for i, test_size in enumerate(x_axis_vals):
            plt.rcParams.update({'font.size': font_size // 2})
            rel_axis.text(test_size, rel_res[i], f'{rel_res[i] * 100:.1f}%')
            plt.rcParams.update({'font.size': font_size})
        rel_axis.set_xticks(x_axis_vals)
        rel_axis.set_xlabel(xlabel)
        rel_axis.set_yticks(np.arange(0, 1.01, 0.1))
        rel_axis.set_ylabel('scores')
    if one_plot:
        axs.legend()

    if save_plot:
        save_plot_func(title=title, data=results)

    plt.show()
    plt.rcParams.update({'font.size': font_size})  # Make sure to revert to given font size
