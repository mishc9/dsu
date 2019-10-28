import os
from pathlib import Path
from typing import Union, Callable, Tuple, Optional

import matplotlib.pyplot as plt
import pandas as pd

from dsu.fsutils import make_dir_safely

# Todo: parallelize plotters with Pool
from dsu.vis.plotters import boxplot, hist, plot, td_heatmap, null_frequency, plotter_wrapper
from dsu.vis.utils import _get_subplot_grid_params
import yaml


def load_data_description(conf_path: Union[str, Path]) -> Optional[dict]:
    try:
        with open(conf_path, 'r') as stream:
            try:
                parsed = yaml.safe_load(stream)
                return parsed
            except yaml.YAMLError as exc:
                print(exc)
                return None
    except FileNotFoundError:
        return None


def visualize(dataframe: pd.DataFrame,
              path_to_file: Union[str, Path],
              file_suffix=None,
              plotters: Tuple = (hist, boxplot, plot, td_heatmap, null_frequency,),
              plotter_config: Optional[Union[str, Path]] = None
              ):
    if plotter_config is not None:
        plotter_config = load_data_description(plotter_config)
    for plotter in plotters:
        name = plotter.__name__
        print(f'Going for a {name}')
        use_plotter(dataframe, path_to_file, plotter, file_suffix,
                    plotter_config.get(name, None)
                    if plotter_config is not None
                    else None)


def use_plotter(dataframe: pd.DataFrame,
                figsave_path: Union[str, Path],
                plotter: Callable,
                file_suffix=None,
                plotter_config=None
                ):
    cols = dataframe.columns
    col_iter = iter(cols)
    if plotter_config is None:
        plotter_config = dict()

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
                    plotter_wrapper(plotter, axes[n_row, n_col], dataframe[col], **plotter_config)
                except TypeError:
                    plotter_wrapper(plotter, axes, dataframe[col], **plotter_config)
                    break
            except StopIteration:
                break
    make_dir_safely(figsave_path)
    fig.tight_layout()
    if file_suffix is None:
        file_suffix = ''
    else:
        file_suffix = '_' + file_suffix
    print(f"--> saving image {str(figsave_path)}")
    fig.savefig(os.path.join(figsave_path, plotter.__name__ + file_suffix + '.png'))
