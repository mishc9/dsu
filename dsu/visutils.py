import os
from math import ceil, floor
from pathlib import Path
from typing import Union, Callable, Optional, Tuple, Sized

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from dsu.fsutils import make_dir_safely


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


def n_bins(series: pd.Series, default: int = 25, max_bins: int = 100) -> int:
    """
    Calculate number of bins on histogram via Freedman–Diaconis rule
    :param max_bins: int, max. number of bins
    :param default: int, default number of bins
    :param series: pandas series
    :return: int, n of bins
    """
    width = bin_width(series)
    dist = max(series) - min(series)
    val = (ceil(dist / width)
           if width is not None
              and not np.isclose(width, 0)
           else default)
    return val if val <= max_bins else max_bins


def _get_subplot_grid_params(cols):
    sqrt = floor(cols ** (0.5))
    n_cols = max(1, floor(cols / sqrt))
    n_rows = ceil(cols / sqrt)
    return n_rows, n_cols


def _get_freq(dataframe: Sized, max_points: int = 25000) -> int:
    freq = ceil(len(dataframe) / max_points)
    return freq


def violin(ax: Axes, dataframe, col, max_points: int = 25000):
    # To slow if print each row, so we're using slices
    # Todo: this function is broken and don't do anything
    freq = floor(len(dataframe) / max_points)
    return ax.violinplot(dataframe[col].iloc[::_get_freq(dataframe, max_points)])


def boxplot(ax: Axes, dataframe, col):
    """
    Do a boxplot
    :param ax:
    :param dataframe:
    :param col:
    :return:
    """
    ser: pd.Series = dataframe[col]
    ser: pd.Series = ser[(ser > ser.quantile(q=0.01)) & (ser < ser.quantile(q=0.99))]
    ax.set_title(col)
    return ser.plot.box(ax=ax)


def hist(ax: Axes, dataframe: pd.DataFrame, col, drop_outliers: bool = True):
    """
    Plot histogram
    :param ax:
    :param dataframe:
    :param col:
    :param drop_outliers:
    :return:
    """
    ser: pd.Series = dataframe[col]
    if drop_outliers:
        ser: pd.Series = ser[(ser > ser.quantile(q=0.01)) & (ser < ser.quantile(q=0.99))]
    ax.set_title(col)
    return ser.hist(ax=ax, bins=n_bins(ser))


def plot(ax: Axes, dataframe: pd.DataFrame, col, max_points: int = 25000):
    """
    Plot simple curve
    :param ax:
    :param dataframe:
    :param col:
    :param max_points:
    :return:
    """
    # To slow if we'll print each row, so we use slices
    ser: pd.Series = dataframe[col].iloc[::_get_freq(dataframe, max_points)]
    return ser.plot(ax=ax)


def td_heatmap(ax: Axes, dataframe: pd.DataFrame, col, n_segments=1000):
    """
    Time-distributed heatmap of parameters (aka proxy for time-distributed Joy Division-like histogram)
    :param ax:
    :param dataframe:
    :param col:
    :param n_segments:
    :return:
    """
    X = []
    Y = []
    h = []
    ax.pcolormesh(X, Y, h)


def null_frequency(ax: Axes, dataframe: pd.DataFrame, col, fs=60):
    """
    Plot spectrogram of zero frequency
    :param ax:
    :param dataframe:
    :param col:
    :return:
    """
    series: pd.Series = dataframe[col]
    null_series: pd.Series = series.isna().astype(int)
    values = null_series.values
    #
    ax.specgram(values, NFFT=2**10, Fs=fs)


def visualize(dataframe: pd.DataFrame,
              path_to_file: Union[str, Path],
              file_suffix=None,
              plotters: Tuple = (hist, boxplot, plot,)):
    for plotter in plotters:
        print(f'Going for a {plotter.__name__}')
        use_plotter(dataframe, path_to_file, plotter, file_suffix)


def use_plotter(dataframe: pd.DataFrame,
                figsave_path: Union[str, Path],
                plotter: Callable,
                file_suffix=None,
                ):
    cols = dataframe.columns
    col_iter = iter(cols)

    n_rows, n_cols = _get_subplot_grid_params(len(cols))
    fig, axes = plt.subplots(nrows=n_rows,
                             ncols=n_cols,
                             figsize=(25, 25))
    for n_row in range(n_rows):
        for n_col in range(n_cols):
            try:
                col = next(col_iter)
                print(f'Plotting {col}')
                try:
                    plotter(axes[n_row, n_col], dataframe, col)
                except TypeError:
                    plotter(axes, dataframe, col)
                    break
            except StopIteration:
                break
    make_dir_safely(figsave_path)
    fig.tight_layout()
    if file_suffix is None:
        file_suffix = ''
    else:
        file_suffix = '_' + file_suffix
    fig.savefig(os.path.join(figsave_path, plotter.__name__ + file_suffix + '.png'))
