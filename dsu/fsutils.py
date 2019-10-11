import os
from pathlib import Path
from typing import Union, Iterable


def traverse_subdirectories(path: Union[Path, str],
                            pattern=None,
                            filter_files_only=True) -> Iterable[Path]:
    """

    :param path: path to root dir
    :param pattern: pattern if we need some filtering
    :param filter_files_only: don't filter directories with pattern if True
    :return:
    """
    path = Path(path)
    if path.is_dir():
        if pattern is None:
            s = [traverse_subdirectories(p) for p in path.iterdir()]
        else:
            s = [traverse_subdirectories(p) for p in path.glob(pattern)]
            if filter_files_only:
                s_dirs = [traverse_subdirectories(p) for p in path.iterdir() if p.is_dir()]
                s += s_dirs
        return [i for x in s for i in x]
    else:
        return [path]


def make_dir_safely(path: Union[Path, str]) -> None:
    """
    Make directory iif it not exists
    :param path: path to directory
    :return:
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != os.errno.EEXIST:
            raise
