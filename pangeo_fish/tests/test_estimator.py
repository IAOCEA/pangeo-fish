import numpy as np
import pandas as pd
import pytest
import xarray as xr
from tlz.functoolz import curry

from pangeo_fish.hmm.estimator import EagerEstimator
from pangeo_fish.hmm.optimize import EagerBoundsSearch
from pangeo_fish.hmm.prediction import Gaussian1DHealpix


@pytest.fixture
def sample_dataset():
    """
    Create a sample xarray dataset with random probabilities, initial/final states, and a mask.
    """
    num_time_steps = 10
    start_time = pd.Timestamp("2022-06-13T12:00:00")
    end_time = pd.Timestamp("2022-06-24T05:00:00")
    time = pd.date_range(start=start_time, end=end_time, periods=num_time_steps)
    nside = 8
    cell_ids = np.arange(4 * nside**2, 6 * nside**2)
    num_cells = cell_ids.size
    initial = np.zeros(num_cells)
    initial[len(initial) // 2] = 0.80
    initial[len(initial) // 2 + 1] = 0.20
    final = np.zeros(num_cells)
    final[(len(final) // 2)] = 0.75
    mask = np.ones(num_cells)
    pdf = np.random.rand(num_time_steps, num_cells)
    pdf /= pdf.sum(axis=1, keepdims=True)
    ds = xr.Dataset(
        coords={"cell_ids": ("cells", cell_ids), "time": ("time", time)},
        data_vars={
            "pdf": (("time", "cells"), pdf),
            "initial": ("cells", initial),
            "final": ("cells", final),
            "mask": ("cells", mask),
        },
    )
    return ds.dggs.decode(
        {"grid_name": "healpix", "level": 12, "indexing_scheme": "nested"}
    ).dggs.assign_latlon_coords()


@pytest.fixture
def predictor_factory(sample_dataset):
    """
    Return a configured Gaussian1DHealpix predictor factory using sample dataset parameters.
    """
    return curry(
        Gaussian1DHealpix,
        cell_ids=sample_dataset["cell_ids"].data,
        grid_info=sample_dataset.dggs.grid_info,
        truncate=4.0,
        weights_threshold=1e-8,
        pad_kwargs={"mode": "constant", "constant_value": 0},
        optimize_convolution=True,
    )


@pytest.mark.parametrize("sigma", [0.0004, 0.0002])
def test_eager_estimator_score(sample_dataset, predictor_factory, sigma):
    """
    Test that the score method returns a float and meets expected criteria for given sigma values.
    """
    estimator = EagerEstimator(predictor_factory=predictor_factory, sigma=None)
    score = estimator.set_params(sigma=sigma).score(sample_dataset)
    assert isinstance(score, float), "Score should be a float."
    assert (
        score >= sample_dataset.sizes["time"]
    ), "Score should be equal to or bigger than the number of timesteps."


@pytest.mark.parametrize("sigma", [0.0001, 0.01, 0.005, 0.02])
def test_eager_estimator_predict_proba(sample_dataset, predictor_factory, sigma):
    """
    Test that `predict_proba` computes positive state probabilities summing to 1 for each time step.
    """
    estimator = EagerEstimator(predictor_factory=predictor_factory, sigma=sigma)
    state_probabilities = estimator.predict_proba(sample_dataset)
    print(state_probabilities[9])
    assert (
        state_probabilities > 0
    ).sum() > 0, "Probabilities should contain valid positive values."
    assert np.allclose(
        state_probabilities.sum(axis=1), 1
    ), "The sum for each time step should be 1."


@pytest.mark.parametrize("bounds", [(1e-4, 1), (1, 5), (0.1, 10)])
@pytest.mark.parametrize("tolerance", [1e-3, 1e-6])
def test_eager_bounds_search(sample_dataset, predictor_factory, bounds, tolerance):
    """
    Test that the EagerBoundsSearch optimizer instance is created correctly with given bounds and tolerance.
    """
    estimator = EagerEstimator(sigma=None, predictor_factory=predictor_factory)
    optimizer = EagerBoundsSearch(
        estimator, bounds, optimizer_kwargs={"disp": 3, "xtol": tolerance}
    )
    assert isinstance(optimizer, EagerBoundsSearch)


@pytest.mark.parametrize("tolerance", [1e-3, 1e-4])
def test_fit_sigma(sample_dataset, predictor_factory, tolerance):
    """
    Test that the `fit` method of EagerBoundsSearch finds a valid sigma without NaN values.
    """
    estimator = EagerEstimator(sigma=None, predictor_factory=predictor_factory)
    optimizer = EagerBoundsSearch(
        estimator,
        (1e-4, 0.0004905038455501491),
        optimizer_kwargs={"disp": 3, "xtol": tolerance},
    )
    optimized = optimizer.fit(sample_dataset)
    print("Type of optimized:", type(optimized))
    if hasattr(optimized, "sigma"):
        result = optimized.sigma
        if isinstance(result, np.ndarray | float | int):
            assert not np.isnan(result).any(), "Optimized sigma is nan"
        else:
            raise ValueError("Optimized result is not in the expected format ")
