from math import ceil, floor
from typing import Optional, Sized

import numpy as np
import pandas as pd


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
    sqrt = ceil(cols ** (0.5))
    n_cols = sqrt
    n_rows = ceil(cols / sqrt) if cols % sqrt != 0 else cols // sqrt
    return n_rows, n_cols


def _get_freq(dataframe: Sized, max_points: int = 25000) -> int:
    freq = ceil(len(dataframe) / max_points)
    return freq
