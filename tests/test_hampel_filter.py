import pytest
import numpy as np
import pandas as pd
from src.hampel_filter import hampel_filter


@pytest.fixture
def sample_outlier_indices():
    return [50, 150]


@pytest.fixture
def sample_data(sample_outlier_indices):
    x = np.linspace(-np.pi, np.pi, 201)
    sin_x = np.sin(x)
    sin_x[sample_outlier_indices] = 0  # as outliers
    return sin_x


def test_np_input(sample_data, sample_outlier_indices):
    assert all(hampel_filter(sample_data) == np.array(sample_outlier_indices))


def test_list_input(sample_data, sample_outlier_indices):
    assert hampel_filter(list(sample_data)) == sample_outlier_indices


def test_series_input(sample_data, sample_outlier_indices):
    """
    When timeseries of type pd.Series is given, the returned indices must much the Series object's index values.
    """
    sr_sample_data = pd.Series(sample_data)
    sr_sample_data.index = sr_sample_data.index + 10
    outlier_indices = hampel_filter(sr_sample_data)
    assert all(outlier_indices == sr_sample_data[outlier_indices].index)


def test_str_input():
    with pytest.raises(ValueError):
        hampel_filter("[1, 2, 3]")


def test_negative_window_size(sample_data):
    with pytest.raises(ValueError):
        hampel_filter(sample_data, window_size=-1)


def test_zero_window_size(sample_data):
    with pytest.raises(ValueError):
        hampel_filter(sample_data, window_size=0)


def test_str_window_size(sample_data):
    with pytest.raises(ValueError):
        hampel_filter(sample_data, window_size="5")


def test_float_window_size(sample_data):
    with pytest.raises(ValueError):
        hampel_filter(sample_data, window_size=3.0)


def test_negative_n_sigma(sample_data):
    with pytest.raises(ValueError):
        hampel_filter(sample_data, n_sigma=-1)


def test_str_n_sigma(sample_data):
    with pytest.raises(ValueError):
        hampel_filter(sample_data, n_sigma="3")


def test_str_n_sigma(sample_data):
    with pytest.raises(ValueError):
        hampel_filter(sample_data, n_sigma="3")

