from pathlib import Path
from typing import Union, Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from dsu.fsutils import make_dir_safely
from math import ceil, floor


def bin_width(ser: pd.Series) -> Optional[int]:
    """
    Helper for a Freedman-Diaconis method
    :param ser: pd.Series to binarize
    :return: int, # of bins
    """
    q25 = ser.quantile(q=0.25)
    q75 = ser.quantile(q=0.75)
    iqr = q75 - q25
    quot = (len(ser) ** (1 / 3))
    if np.isclose(quot, 0):
        return None
    else:
        return 2 * iqr / quot


def n_bins(ser: pd.Series, default: int = 25, max_bins: int = 500) -> int:
    """
    Calculate number of bins on histogram via Freedman–Diaconis rule
    :param max_bins: int, max. number of bins
    :param default: int, default number of bins
    :param ser: pandas series
    :return: int, n of bins
    """
    width = bin_width(ser)
    dist = max(ser) - min(ser)
    val = (ceil(dist / width)
           if width is not None
              and not np.isclose(width, 0)
           else default)
    return val if val <= max_bins else max_bins


def make_subplots(cols):
    sqrt = floor(cols ** (0.5))
    n_cols = floor(cols / sqrt)
    n_rows = ceil(cols / sqrt)

    return n_rows, n_cols


def violin(ax, dataframe, col):
    # To slow if print each row, so we're using slices
    # Todo: this function is broken ant don't plot anything
    return ax.violinplot(dataframe[col].iloc[::100])


def boxplot(ax, dataframe, col):
    ser: pd.Series = dataframe[col]
    ser: pd.Series = ser[(ser > ser.quantile(q=0.01)) & (ser < ser.quantile(q=0.99))]
    ax.set_title(col)
    return ser.plot.box(ax=ax)  # n_bins(ser))


def hist(ax, dataframe: pd.DataFrame, col):
    ser: pd.Series = dataframe[col]
    ser = ser[(ser > ser.quantile(q=0.01)) & (ser < ser.quantile(q=0.99))]
    ax.set_title(col)
    return ser.hist(ax=ax, bins=25)  # n_bins(ser))


def plot(ax, dataframe: pd.DataFrame, col):
    # To slow if we'll print each row, so we're using slices
    ser: pd.Series = dataframe[col].iloc[::100]
    return ser.plot(ax=ax)


def visualize(dataframe: pd.DataFrame,
              path_to_file: Union[str, Path],
              file_suffix=None,
              plotters: Tuple = (hist, boxplot, plot,)):
    for plotter in plotters:
        print(f'Going for a {plotter.__name__}')
        plot_somehow(dataframe, path_to_file, plotter, file_suffix)


def plot_somehow(dataframe: pd.DataFrame,
                 figsave_path: Union[str, Path],
                 plotter: Callable,
                 file_suffix=None,
                 ):
    cols = dataframe.columns
    col_iter = iter(cols)

    n_rows, n_cols = make_subplots(len(cols))
    fig, axes = plt.subplots(nrows=n_rows,
                             ncols=n_cols,
                             figsize=(25, 25))
    for n_row in range(n_rows):
        for n_col in range(n_cols):
            try:
                col = next(col_iter)
                print(f'Plotting {col}')
                plotter(axes[n_row, n_col], dataframe, col)
            except StopIteration:
                break
    make_dir_safely(figsave_path)
    fig.tight_layout()
    fig.savefig(os.path.join(figsave_path, plotter.__name__ + '_' + file_suffix or '' + '.png'))
