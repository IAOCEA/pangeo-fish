import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pangeo_fish.hmm.decode import mean_track, modal_track, viterbi, viterbi2


def generate_test_emission():
    """enerate small dataset"""
    y_size, x_size, time_size = 10, 10, 5
    lat_start, lon_start = 42.45, -7.398

    lat = np.linspace(lat_start, lat_start + 0.01 * (y_size - 1), y_size)
    lon = np.linspace(lon_start, lon_start + 0.01 * (x_size - 1), x_size)
    time = pd.date_range("2022-06-13T12:00:00", periods=time_size, freq="D")

    latitude, longitude = np.meshgrid(lat, lon, indexing="ij")

    final = np.zeros((y_size, x_size))
    initial = np.zeros((y_size, x_size))
    mask = np.full((y_size, x_size), 1.0)
    pdf = np.full((time_size, x_size, y_size), 2.0)
    cell_ids = np.arange(y_size * x_size).reshape((y_size, x_size))

    dataset = xr.Dataset(
        {
            "final": (("y", "x"), final),
            "initial": (("y", "x"), initial),
            "mask": (("y", "x"), mask),
            "pdf": (("time", "x", "y"), pdf),
        },
        coords={
            "y": np.arange(y_size),
            "x": np.arange(x_size),
            "time": time,
            "latitude": (("y", "x"), latitude),
            "longitude": (("y", "x"), longitude),
            "cell_ids": (("y", "x"), cell_ids),
        },
    )

    return dataset


@pytest.fixture
def shared_dataset():
    """Generate states matrices"""
    time_size, y_size, x_size = 5, 3, 4
    times = pd.date_range("2022-06-13T12:00:00", periods=time_size, freq="D")
    latitudes = np.linspace(42.45, 42.68, y_size * x_size).reshape(y_size, x_size)
    longitudes = np.linspace(-7.398, -7.375, y_size * x_size).reshape(y_size, x_size)
    cell_ids = np.arange(1, y_size * x_size + 1).reshape(y_size, x_size)

    raw_states = np.random.rand(time_size, y_size, x_size)
    states = raw_states / raw_states.sum(
        axis=(1, 2), keepdims=True
    )  # Normalize per time step

    states = xr.DataArray(states, dims=("time", "y", "x"), name="states")

    return xr.Dataset(
        {"states": states},
        coords={
            "time": times,
            "latitude": (("y", "x"), latitudes),
            "longitude": (("y", "x"), longitudes),
            "cell_ids": (("y", "x"), cell_ids),
        },
    )


def test_mean_track(shared_dataset):
    """Test mean_track with a small dataset"""

    result = mean_track(shared_dataset)
    print(result)


@pytest.mark.parametrize("coords", [["latitude", "longitude"]])
def test_modal_track(shared_dataset, coords):
    """Test modal_track with a small dataset"""

    result = modal_track(shared_dataset, coords=coords)
    print(result)


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_viterbi(sigma):
    """Test viterbi with a small dataset"""
    emission = generate_test_emission()
    result = viterbi(emission, sigma)
    print(result)

    assert isinstance(result, xr.Dataset)
    for coord in ["time", "latitude", "longitude"]:
        assert coord in result.coords, f"Coordinate '{coord}' not found in the result."

    assert result.coords["time"].size == generate_test_emission().coords["time"].size


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_viterbi2(shared_dataset, sigma):
    """Test viterbi2 with a small dataset"""
    emission = generate_test_emission()
    result = viterbi2(emission, sigma)
    print(result)

    assert isinstance(result, xr.Dataset)
    for coord in ["time", "latitude", "longitude"]:
        assert coord in result.coords, f"Coordinate '{coord}' not found in the result."

    assert result.coords["time"].size == generate_test_emission().coords["time"].size
