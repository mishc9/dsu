from functools import partial
from typing import Iterable, Callable, Union

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from dsu.vis.utils import n_bins, _get_freq


def _drop_if(series: pd.Series, dropna: bool) -> pd.Series:
    return series.dropna() if dropna else series


def plotter_wrapper(plotter: Callable,
                    ax: Axes,
                    series: pd.Series,
                    dropna: bool = False,
                    **kwargs):
    try:
        return plotter(ax, _drop_if(series, dropna), **kwargs)
    except Exception as e:
        print(str(e))
        return None


def violin(ax: Axes, series: pd.Series, max_points: int = 25000):
    # To slow if we'll print each row, so we're using slices
    # Todo: this function is broken and don't do anything
    return ax.violinplot(series.iloc[::_get_freq(series, max_points)])


def boxplot(ax: Axes, series: pd.Series, drop_outliers: bool = True):
    """
    Do a boxplot
    :param ax:
    :param series:
    :param series.name:
    :return:
    """
    if drop_outliers:
        series: pd.Series = series[(series > series.quantile(q=0.01)) & (series < series.quantile(q=0.99))]
    ax.set_title(series.name)
    return series.plot.box(ax=ax)


def hist(ax: Axes, series: pd.Series, drop_outliers: bool = True, lo=0.01, hi=0.99):
    """
    Plot histogram
    :param hi:
    :param lo:
    :param ax:
    :param series:
    :param series.name:
    :param drop_outliers:
    :return:
    """
    if drop_outliers:
        series: pd.Series = series[(series > series.quantile(q=lo)) & (series < series.quantile(q=hi))]
    ax.set_title(series.name)
    return series.hist(ax=ax, bins=n_bins(series))


def plot(ax: Axes, series: pd.Series, max_points: int = 25000):
    """
    Plot simple curve
    :param ax:
    :param series:
    :param series.name:
    :param max_points:
    :return:
    """
    # To slow if we'll print each row, so we use slices
    ser: pd.Series = series.iloc[::_get_freq(series, max_points)]
    ax.set_title(series.name)
    return ser.plot(ax=ax)


def reindex_date(data: Union[pd.Series, pd.DataFrame], freq) -> Union[pd.Series, pd.DataFrame]:
    start, end = data.index.min(), data.index.max()
    DATE_RANGE = pd.date_range(start, end, freq=freq)
    print(f'Setting date range {start}:{end}, freq {freq}')
    return data.reindex(DATE_RANGE)


def null_frequency(ax: Axes, series: pd.Series, freq='1H'):
    """
    Plot spectrogram of zero frequency
    :param ax:
    :param series:
    :param series.name:
    :return:
    """
    # Todo: smart selection of frequency
    series: pd.Series = reindex_date(series, freq=freq)
    null_series: pd.Series = series.notna().astype(int)
    groups = null_series.groupby(pd.Grouper(freq=freq)).sum()
    groups /= pd.to_timedelta(freq).total_seconds()

    ax.set_title(f"Not Null freq. of {series.name}")
    return ax.plot(groups)


def td_heatmap(ax: Axes,
               series: pd.Series,
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
    :param series:
    :param series.name:
    :param n_segments:
    :return:
    """

    # Todo: make proper x-axis label

    def hist_of_group(series: pd.Series, bins: Iterable):
        hist_values, _ = np.histogram(series.values, bins)
        return hist_values

    ax.set_title(f"Trend of {series.name}")
    series: pd.Series = reindex_date(series, freq=freq)
    if drop_outliers:
        series: pd.Series = series[(series > series.quantile(q=lo)) & (series < series.quantile(q=hi))]
    min_val, max_val = series.min(), series.max()
    number_of_bins = n_bins(series)
    if number_of_bins == 0:
        number_of_bins = 25
    try:
        bins = np.arange(min_val, max_val, (max_val - min_val) / number_of_bins)
        groups: pd.Series = series.groupby(pd.Grouper(freq=freq)).apply(
            partial(hist_of_group, bins=bins))

        # How X, Y, C work:
        #
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
    except ValueError:
        print("Failed to plot!")
