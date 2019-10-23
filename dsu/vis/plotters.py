from functools import partial
from math import floor
from typing import Iterable

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from dsu.vis.utils import n_bins, _get_freq


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


def hist(ax: Axes, dataframe: pd.DataFrame, col, drop_outliers: bool = True, lo=0.01, hi=0.99):
    """
    Plot histogram
    :param hi:
    :param lo:
    :param ax:
    :param dataframe:
    :param col:
    :param drop_outliers:
    :return:
    """
    ser: pd.Series = dataframe[col]
    if drop_outliers:
        ser: pd.Series = ser[(ser > ser.quantile(q=lo)) & (ser < ser.quantile(q=hi))]
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


def td_heatmap(ax: Axes,
               dataframe: pd.DataFrame,
               col,
               freq='1H',
               drop_outliers: bool = True,
               lo: float = 0.01,
               hi: float = 0.99):

    """
    Time-distributed heatmap of parameters (aka proxy for time-distributed Joy Division-like histogram)
    :param hi:
    :param lo:
    :param drop_outliers:
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
    # Always filter outliers
    if drop_outliers:
        series: pd.Series = series[(series > series.quantile(q=lo)) & (series < series.quantile(q=hi))]
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
