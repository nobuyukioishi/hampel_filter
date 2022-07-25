import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from typing import Union, List


def hampel_filter(x: Union[List, pd.Series, np.ndarray], window_size: int = 5, n_sigma: int = 3, c: float = 1.4826) \
        -> np.array:
    """ Outlier detection using the Hampel identifier

    :param x: timeseries values of type List, numpy.ndarray, or pandas.Series
    :param window_size: length of the sliding window, a positive odd integer.
        (`window_size` - 1) // 2 adjacent samples on each side of the current sample are used for calculating median.
    :param n_sigma: threshold for outlier detection, a real scalar greater than or equal to 0. default is 3.
    :param c: consistency constant. default is 1.4826, supposing the given timeseries values are normally distributed.
    :return: the outlier indices
    """

    if not (type(x) == List or type(x) == np.ndarray or type(x) == pd.Series):
        raise ValueError("x must be either of type List, numpy.ndarray, or pandas.Series.")

    if not (type(window_size) == int and window_size % 2 == 1 and window_size > 0):

        raise ValueError("window_size must be a positive odd integer greater than 0.")

    # convert the timeseries values to numpy.array
    np_x = np.array(x)

    np_x_windows = sliding_window_view(np_x, window_shape=window_size)
    rolling_median = np.median(np_x_windows, axis=1)
    rolling_sigma = c * np.median(np.abs(np_x_windows - rolling_median.reshape(-1, 1)), axis=1)

    outlier_indices = np.nonzero(np.abs(np_x[(window_size-1)//2:-(window_size-1)//2] - rolling_median)
                                 >= (n_sigma * rolling_sigma))[0] + (window_size-1)//2

    if type(x) == List:
        # When x is of List[float | int], return the indices in List.
        return list(outlier_indices)
    elif type(x) == pd.Series:
        # When x is of pd.Series, return the indices of the Series object.
        return x.index[outlier_indices]
    else:
        return outlier_indices
