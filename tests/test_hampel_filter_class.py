import pytest
import numpy as np
import pandas as pd
from src.hampel_filter import HampelFilter


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
    hampel = HampelFilter()
    assert all(hampel.apply(sample_data).get_indices() == np.array(sample_outlier_indices))


def test_list_input(sample_data, sample_outlier_indices):
    hampel = HampelFilter()
    assert hampel.apply(list(sample_data)).get_indices() == sample_outlier_indices


def test_series_input(sample_data, sample_outlier_indices):
    """
    When timeseries of type pd.Series is given, the returned indices must much the Series object's index values.
    """
    hampel = HampelFilter()
    sr_sample_data = pd.Series(sample_data)
    sr_sample_data.index = sr_sample_data.index + 10
    outlier_indices = hampel.apply(sr_sample_data).get_indices()
    assert all(outlier_indices == sr_sample_data[outlier_indices].index)


def test_get_indices_before_apply():
    hampel = HampelFilter()
    with pytest.raises(AttributeError):
        hampel.get_indices()


def test_get_boundaries_before_apply():
    hampel = HampelFilter()
    with pytest.raises(AttributeError):
        hampel.get_boundaries()


def test_str_input():
    with pytest.raises(ValueError):
        hampel = HampelFilter()
        hampel.apply("[1, 2, 3]")


def test_negative_window_size():
    with pytest.raises(ValueError):
        HampelFilter(window_size=-1)


def test_zero_window_size():
    with pytest.raises(ValueError):
        HampelFilter(window_size=0)


def test_str_window_size():
    with pytest.raises(ValueError):
        HampelFilter(window_size="5")


def test_float_window_size():
    with pytest.raises(ValueError):
        HampelFilter(window_size=3.0)


def test_negative_n_sigma():
    with pytest.raises(ValueError):
        HampelFilter(n_sigma=-1)


def test_str_n_sigma():
    with pytest.raises(ValueError):
        HampelFilter(n_sigma="3")


def test_str_n_sigma():
    with pytest.raises(ValueError):
        HampelFilter(n_sigma="3")

