import numpy as np
import xarray as xr

from pangeo_fish.grid import center_longitude


def test_center_longitude():
    num_cells = 10  # Small grid size for testing
    center = 0

    lons = np.linspace(-20, 380, num_cells)
    lats = np.linspace(-90, 90, num_cells)

    ds = xr.Dataset(
        coords={
            "longitude": ("cells", lons),
            "latitude": ("cells", lats),
        }
    )
    actual = center_longitude(ds)

    lower, upper = -180, 180

    assert (
        (actual["longitude"] >= lower) & (actual["longitude"] <= upper)
    ).all(), "Longitudes are not properly centered."
