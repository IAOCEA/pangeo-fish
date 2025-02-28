"""Miscellaneous operations around biologging data."""

import numpy as np
import pandas as pd
import xarray as xr


def to_time_slice(times):
    subset = times.where(times.notnull(), drop=True)

    min_ = subset.isel(event_name=0)
    max_ = subset.isel(event_name=1)

    return slice(min_.data, max_.data)


def adapt_model_time(slice_):
    start = np.datetime64(slice_.start)
    stop = np.datetime64(slice_.stop)

    # only if [minute, sec] part of `slice_.start` < 30:00
    if pd.Timestamp(start).minute < 30:
        model_start = start - np.timedelta64(30, "m")
    else:
        model_start = start

    # only if [minute, sec] part of `slice_.stop` > 30:00
    if pd.Timestamp(stop).minute > 30:
        model_stop = stop + np.timedelta64(30, "m")
    else:
        model_stop = stop

    return slice(model_start, model_stop)


def reshape_by_bins(ds, *, dim, bins, other_dim="obs"):
    def expand_group(group, *, dim, other_dim):
        return (
            group.rename_dims({dim: other_dim})
            .drop_indexes(dim)
            .assign_coords({other_dim: lambda ds: ds[other_dim]})
        )

    index = bins.to_index()
    grouper = xr.groupers.BinGrouper(index, include_lowest=True)

    processed = ds.groupby({dim: grouper}).map(
        expand_group, dim=dim, other_dim=other_dim
    )

    return (
        processed.drop_vars("time")
        .rename_dims({"time_bins": "time"})
        .rename_vars({"time_bins": "time"})
    )
