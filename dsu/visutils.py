import os
from functools import partial
from math import ceil, floor
from pathlib import Path
from typing import Union, Callable, Optional, Tuple, Sized, Iterable

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
    if bin_width is None:
        return default
    dist = max(series) - min(series)
    try:
        val = (ceil(dist / width)
               if width is not None
                  and not np.isclose(width, 0)
               else default)
    except ValueError:
        return default
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
    ax.set_title(col)
    return ser.plot(ax=ax)


def null_frequency(ax: Axes, dataframe: pd.DataFrame, col, freq='1H'):
    """
    Plot spectrogram of zero frequency
    :param ax:
    :param dataframe:
    :param col:
    :return:
    """
    # Todo: smart selection of frequency
    series: pd.Series = dataframe[col]
    null_series: pd.Series = series.notna().astype(int)
    groups = null_series.groupby(pd.Grouper(freq=freq)).sum()
    groups /= pd.to_timedelta(freq).total_seconds()

    ax.set_title(f"Not Null freq. of {col}")
    return ax.plot(groups)


def td_heatmap(ax: Axes, dataframe: pd.DataFrame, col, freq='1H'):
    """
    Time-distributed heatmap of parameters (aka proxy for time-distributed Joy Division-like histogram)
    :param freq:
    :param ax:
    :param dataframe:
    :param col:
    :param n_segments:
    :return:
    """

    # Todo: make proper x-axis label

    def hist_of_group(series: pd.Series, bins: Iterable):
        hist_values, _ = np.histogram(series.values, bins)
        return hist_values

    ax.set_title(f"Trend of {col}")
    series: pd.Series = dataframe[col]
    min_val, max_val = series.min(), series.max()
    number_of_bins = n_bins(series)
    if number_of_bins == 0:
        number_of_bins = 25
    bins = np.arange(min_val, max_val, (max_val - min_val) / number_of_bins)
    groups: pd.Series = series.groupby(pd.Grouper(freq=freq)).apply(
        partial(hist_of_group, bins=bins))

    # How X, Y, C work:
    # (X[i+1, j], Y[i+1, j])      (X[i+1, j+1], Y[i+1, j+1])
    #                     +--------+
    #                     | C[i,j] |
    #                     +--------+
    # (X[i, j], Y[i, j])          (X[i, j+1], Y[i, j+1]),

    X = np.tile(np.arange(len(groups)), (len(bins), 1)).T
    Y = np.tile(bins, (len(groups), 1))

    def pad(x, size):
        vec = np.zeros(size)
        vec[:len(x)] = x
        return x

    C = np.array([pad(val, len(bins))
                  for val
                  in groups.values])

    ax.pcolormesh(X, Y, C)


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
