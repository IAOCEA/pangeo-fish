import numpy as np
import xarray as xr

from pangeo_fish.diff import diff_z, diff_z_numba


def test_diff_z_numba():
    """
    Test diff_z_numba with consistent input data so that it does not return np.nan.

    Model:
      depth: [0.5, 1.0, 1.5, 2.0, 2.5]
      temp:  [10, 15, 20, 25, 30]

    Tag:
      depth: [1.0, 1.5, 2.0]
      temp:  [14, 19, 24]

    With depth_thresh=0.8, np.max(tag_depth)*depth_thresh = 2.0*0.8 = 1.6.
    Choose bottom = 2.0 (>= 1.6) so the loop computes:
      diff = tag_temp - model_temp_nearest, yielding [-1, -1, -1] → mean = -1.
    """
    model_depth = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
    model_temp = np.array([10, 15, 20, 25, 30], dtype=np.float64)
    tag_depth = np.array([1.0, 1.5, 2.0], dtype=np.float64)
    tag_temp = np.array([14, 19, 24], dtype=np.float64)
    depth_thresh = 0.8
    bottom = 2.0  # bottom >= 2.0*0.8 = 1.6
    result = diff_z_numba(
        model_temp, model_depth, bottom, tag_temp, tag_depth, depth_thresh
    )
    expected = -1.0
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_diff_z():
    """
    Test diff_z via xr.apply_ufunc.

    Model xarray.Dataset:
      - depth: [0.5, 1.0, 1.5, 2.0, 2.5]
      - TEMP: [10, 15, 20, 25, 30]
      - dynamic_depth: same as depth
      - dynamic_bathymetry: 2.5 (>= np.max(pressure)*0.8)
      - TEMP has units "degC"

    Tag xarray.Dataset:
      - temperature: [14, 19, 24]
      - pressure: [1.0, 1.5, 2.0]

    Expected diff is the mean of [-1, -1, -1] → -1.
    """
    depth = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float64)
    temp = np.array([10, 15, 20, 25, 30], dtype=np.float64)
    model = xr.Dataset(
        {
            "TEMP": ("depth", temp),
            "dynamic_depth": ("depth", depth),
            "dynamic_bathymetry": 2.5,
        },
        coords={"depth": depth},
    )
    model["TEMP"].attrs["units"] = "degC"

    tag_temp = np.array([14, 19, 24], dtype=np.float64)
    tag_pressure = np.array([1.0, 1.5, 2.0], dtype=np.float64)
    tag = xr.Dataset(
        {
            "temperature": ("obs", tag_temp),
            "pressure": ("obs", tag_pressure),
        },
        coords={"obs": np.arange(3)},
    )

    result = diff_z(model, tag, depth_threshold=0.8)
    expected = -1.0

    assert isinstance(result, xr.Dataset)
    assert "diff" in result
    np.testing.assert_allclose(result["diff"].values, expected, rtol=1e-5)
    assert result["diff"].attrs["units"] == "degC"
