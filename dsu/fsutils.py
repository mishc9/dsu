import os
from pathlib import Path
from typing import Union, Iterable


def traverse_subdirectories(path: Union[Path, str],
                            pattern=None,
                            filter_files_only=True) -> Iterable[Union[Path, str]]:
    """

    :param path: path to root dir
    :param pattern: pattern if we need some filtering
    :param filter_files_only: don't filter directories with pattern if True
    :return:
    """
    orig_type = type(path)
    path = Path(path)
    if path.is_dir():
        if filter_files_only:
            s = [traverse_subdirectories(p) for p in path.iterdir()]
        else:
            s = [traverse_subdirectories(p) for p in path.iterdir() if p.match(pattern)]
        if pattern is None:
            return [orig_type(i) for x in s for i in x]
        else:
            return [orig_type(i) for x in s for i in x if i.match(pattern)]
    else:
        return [orig_type(path)] if path.match(pattern) else []


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
