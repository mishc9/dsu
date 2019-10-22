# Started test branch
import numpy as np
import pandas as pd

from dsu.visutils import visualize, null_frequency

size = 100000
null_count = 25000
NULL_DATA = pd.DataFrame({'values': range(size)})
array = np.arange(size)
np.random.shuffle(array)
head = array[:null_count]
NULL_DATA.iloc[head, :] = None


def vis_null_frequency(dataframe: pd.DataFrame):
    visualize(dataframe, '.', None, (null_frequency,))


if __name__ == '__main__':
    vis_null_frequency(NULL_DATA)
