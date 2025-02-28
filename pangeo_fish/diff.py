"""Implements diff operations between tags and reference data."""

import numba
import numpy as np
import xarray as xr

_diff_z_signatures = [
    "void(float32[:], float32[:], float32, float32[:], float32[:], float32, float32[:])",
    "void(float64[:], float64[:], float64, float64[:], float64[:], float32, float64[:])",
    "void(float32[:], float32[:], float32, float64[:], float64[:], float32, float64[:])",
    "void(float64[:], float64[:], float64, float32[:], float32[:], float32, float64[:])",
]


@numba.guvectorize(_diff_z_signatures, "(z),(z),(),(o),(o),()->()", nopython=True)
def _diff_z(model_temp, model_depth, bottom, tag_temp, tag_depth, depth_thresh, result):
    if depth_thresh != 0 and bottom < np.max(tag_depth) * depth_thresh:
        result[0] = np.nan
        return

    diff_temp = np.full_like(tag_depth, fill_value=np.nan)
    mask = ~np.isnan(model_depth) & ~np.isnan(model_temp)
    model_depth_ = np.absolute(model_depth[mask])
    if model_depth_.size == 0:
        result[0] = np.nan
        return

    model_temp_ = model_temp[mask]

    for index in range(tag_depth.shape[0]):
        if not np.isnan(tag_depth[index]):
            diff_depth = np.absolute(model_depth_ - tag_depth[index])

            idx = np.argmin(diff_depth)

            diff_temp[index] = tag_temp[index] - np.absolute(model_temp_[idx])

    result[0] = np.mean(diff_temp[~np.isnan(diff_temp)])


def diff_z_numba(model_temp, model_depth, bottom, tag_temp, tag_depth, depth_thresh):
    with np.errstate(all="ignore"):
        # TODO: figure out why the "invalid value encountered" warning is raised
        return _diff_z(
            model_temp, model_depth, bottom, tag_temp, tag_depth, depth_thresh
        )


def diff_z(model, tag, depth_threshold=0.8):
    diff = xr.apply_ufunc(
        diff_z_numba,
        model["TEMP"],
        model["dynamic_depth"],
        model["dynamic_bathymetry"],
        tag["temperature"],
        tag["pressure"],
        kwargs={"depth_thresh": depth_threshold},
        input_core_dims=[["depth"], ["depth"], [], ["obs"], ["obs"]],
        output_core_dims=[[]],
        exclude_dims={},
        vectorize=False,
        dask="parallelized",
        output_dtypes=[model.dtypes["TEMP"]],
    )
    original_units = model["TEMP"].attrs["units"]

    return diff.assign_attrs({"units": original_units}).to_dataset(name="diff")
