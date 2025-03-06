import cf_xarray  # noqa: F401
import pandas as pd
import xarray as xr


def bounds_to_bins(ds, bounds_dim="bounds"):
    variables = {
        name: var for name, var in ds.variables.items() if bounds_dim in var.dims
    }
    bins = {
        f"{name.removesuffix('_bounds')}_bins": xr.Variable(
            [d for d in var.dims if d != bounds_dim],
            pd.IntervalIndex.from_arrays(
                var.isel({bounds_dim: 0}), var.isel({bounds_dim: 1})
            ),
        )
        for name, var in variables.items()
    }

    return ds.assign_coords(bins)
