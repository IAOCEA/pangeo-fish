import numpy as np
import pandas as pd
import pint
import pytest
import xarray as xr
import xdggs  # noqa: F401
from xarray import DataTree

from pangeo_fish.acoustic import emission_probability

# Set up the unit registry
ureg = pint.UnitRegistry()


@pytest.fixture
def dummy_grid():
    """
    Create a minimal DGGS grid dataset with 10 time steps.
    The dataset includes normalized probabilities, initial/final states, and a mask.
    The DGGS chain decodes the grid (Healpix) and assigns lat/lon coordinates.
    """
    num_time_steps = 5
    start_time = pd.Timestamp("2022-06-13T12:00:00")
    end_time = pd.Timestamp("2022-06-24T05:00:00")
    time = pd.date_range(start=start_time, end=end_time, periods=num_time_steps)

    level = 1
    cell_ids = np.arange(4 * 4**level, 6 * 4**level)

    mask = np.full_like(cell_ids, fill_value=True, dtype=bool)
    mask[4:6] = False

    grid_info = {"grid_name": "healpix", "level": level, "indexing_scheme": "nested"}
    ds = xr.Dataset(
        coords={"cell_ids": ("cells", cell_ids, grid_info), "time": ("time", time)},
        data_vars={"mask": ("cells", mask)},
    )
    # Decode the grid using DGGS and assign lat/lon coordinates
    return ds.dggs.decode().dggs.assign_latlon_coords()


@pytest.fixture
def dummy_tag():
    """
    Create a minimal tag DataTree with 4 groups:
      - dst: measured data (e.g., temperature and pressure) along time.
      - tagging_events: event information (e.g., release, fish_death).
      - stations: receiver station metadata.
      - acoustic: acoustic data with detections; the "time" coordinate is kept as a data variable.
    """
    num_time_steps = 10
    times = pd.date_range("2022-06-13T00:00:00", periods=num_time_steps)

    # --- Acoustic group ---
    # Create a dataset with a 'deployment_id' variable.
    # For example, the first 2 time steps get 28689 and the rest get 28690.
    deployment_id_data = np.concatenate(
        (np.repeat(28689, 2), np.repeat(28690, num_time_steps - 2))
    )
    acoustic_ds = xr.Dataset(
        coords={"time": ("time", times)},
        data_vars={"deployment_id": (("time",), deployment_id_data)},
    )

    # --- Tagging events group ---
    event_names = np.array(["release", "fish_death"])
    event_times = np.array(
        [pd.Timestamp("2022-06-13T11:40:30"), pd.Timestamp("2022-06-24T00:00:00")]
    )
    event_longitudes = np.array([-5.098, -5.136])
    event_latitudes = np.array([48.45, 48.43])
    tagging_events_ds = xr.Dataset(
        coords={"event_name": ("event_name", event_names)},
        data_vars={
            "time": (("event_name",), event_times),
            "longitude": (("event_name",), event_longitudes),
            "latitude": (("event_name",), event_latitudes),
        },
    )

    # --- Stations group ---
    # Create a minimal dataset with 2 stations and their metadata.
    deployment_ids = np.array([28689, 28690])
    stations_ds = xr.Dataset(
        coords={"deployment_id": ("deployment_id", deployment_ids)},
        data_vars={
            "station_name": ("deployment_id", ["station_1", "station_2"]),
            "deploy_time": (
                "deployment_id",
                [
                    pd.Timestamp("2022-06-12T12:00:00"),
                    pd.Timestamp("2022-06-12T13:00:00"),
                ],
            ),
            "deploy_longitude": ("deployment_id", [-5.1, -5.12]),
            "deploy_latitude": ("deployment_id", [48.45, 48.46]),
            "recover_time": (
                "deployment_id",
                [
                    pd.Timestamp("2022-06-20T12:00:00"),
                    pd.Timestamp("2022-06-20T13:00:00"),
                ],
            ),
            "recover_longitude": ("deployment_id", [-5.1, -5.12]),
            "recover_latitude": ("deployment_id", [48.45, 48.46]),
        },
    )

    # --- dst group ---
    # Simulate measured data (temperature and pressure) along time.
    temperature = np.random.uniform(15, 25, size=num_time_steps)
    pressure = np.random.uniform(1.0, 2.0, size=num_time_steps)
    dst_ds = xr.Dataset(
        coords={"time": ("time", times)},
        data_vars={
            "temperature": (("time",), temperature),
            "pressure": (("time",), pressure),
        },
    )

    # Assemble the groups into a DataTree
    tag = DataTree.from_dict(
        {
            "dst": dst_ds,
            "tagging_events": tagging_events_ds,
            "stations": stations_ds,
            "acoustic": acoustic_ds,
        },
        name="tag_dummy",
    )
    return tag


@pytest.mark.parametrize("detections", [False, True])
def test_emission_probability(detections, dummy_tag, dummy_grid):
    """
    Test of the ``emission_probability`` function using small datasets (10 time steps).
    """
    # Set receiver buffer size to 1000 m
    receiver_buffer = ureg.Quantity(1000, "m")

    dummy_grid.cell_ids.attrs["lat"] = 0
    dummy_grid.cell_ids.attrs["lon"] = 0

    if not detections:
        del dummy_tag["stations"]
        del dummy_tag["acoustic"]

    actual = emission_probability(
        tag=dummy_tag,
        grid=dummy_grid,
        buffer_size=receiver_buffer,
        nondetections="mask",
        cell_ids="keep",
        chunk_time=24,
        dims=["cells"],
    )

    if not detections:
        expected = xr.Dataset()
        xr.testing.assert_identical(actual, expected)
        return

    assert "acoustic" in actual.data_vars

    actual_ = actual.compute()
    assert actual_["acoustic"].notnull().any()
