import flox.xarray
import healpy as hp
import numpy as np
import pandas as pd
import xarray as xr
from tlz.itertoolz import first
from xarray_healpy.conversions import geographic_to_cartesian
from xarray_healpy.operations import buffer_points

from pangeo_fish import utils
from pangeo_fish.cf import bounds_to_bins
from pangeo_fish.healpy import (
    astronomic_to_cartesian,
    astronomic_to_cell_ids,
    geographic_to_astronomic,
)


def count_detections(detections, by):
    """count the amount of detections by interval

    Parameters
    ----------
    detections : xarray.Dataset
        The detections for a specific tag
    by
        The values to group by. Can be anything that `flox` accepts.

    Returns
    -------
    count : xarray.Dataset
        The counts per interval

    See Also
    --------
    flox.xarray.xarray_reduce
    xarray.Dataset.groupby
    """
    if "bounds" in getattr(by, "dims", []):
        if len(by.cf.bounds) != 1:
            raise ValueError("cannot find a valid bounds variable")

        bounds_var = first(by.cf.bounds.values())[0]
        bins_var = f"{bounds_var.removesuffix('_bounds')}_bins"
        by = bounds_to_bins(by)[bins_var]

    count_on = (
        detections[["deployment_id"]]
        .assign(count=lambda ds: xr.ones_like(ds["deployment_id"], dtype=int))
        .set_coords(["deployment_id"])
    )

    isbin = [False, isinstance(by, pd.IntervalIndex)]

    result = flox.xarray.xarray_reduce(
        count_on,
        "deployment_id",
        "time",
        expected_groups=(None, by),
        isbin=isbin,
        func="sum",
        fill_value=0,
    )

    return result.drop_vars(["time"]).assign_coords({"time": by["time"].variable})


def deployment_reception_masks(
    stations, grid, buffer_size, method="recompute", dims=["x", "y"]
):
    rot = {
        "lat": grid["cell_ids"].attrs.get("lat", 0),
        "lon": grid["cell_ids"].attrs.get("lon", 0),
    }
    if method == "recompute":
        phi, theta = geographic_to_astronomic(
            lon=grid["longitude"], lat=grid["latitude"], rot=rot
        )

        cell_ids = astronomic_to_cell_ids(
            nside=grid.attrs["nside"], theta=theta, phi=phi
        ).assign_attrs(grid["cell_ids"].attrs)

        phi, theta = geographic_to_astronomic(
            lon=stations["deploy_longitude"], lat=stations["deploy_latitude"], rot=rot
        )
        positions = astronomic_to_cartesian(theta=theta, phi=phi, dim="deployment_id")
    elif method == "keep":
        cell_ids = grid["cell_ids"]

        positions = geographic_to_cartesian(
            lon=stations["deploy_longitude"],
            lat=stations["deploy_latitude"],
            rot=rot,
            dim="deployment_id",
        )

    if dims == ["cells"]:
        masks = buffer_points_cells(
            cell_ids,
            positions,
            buffer_size=buffer_size.m_as("m"),
            nside=2 ** cell_ids.attrs["level"],
            factor=2**16,
            intersect=True,
        )
    elif dims == ["x", "y"]:
        masks = buffer_points(
            cell_ids,
            positions,
            buffer_size=buffer_size.m_as("m"),
            nside=2 ** cell_ids.attrs["level"],
            factor=2**16,
            intersect=True,
        )

    return masks.drop_vars(["cell_ids"])


def buffer_points_cells(
    cell_ids,
    positions,
    *,
    buffer_size,
    nside,
    sphere_radius=6371e3,
    factor=4,
    intersect=False,
):
    """ """

    def _buffer_masks(cell_ids, vector, nside, radius, factor=4, intersect=False):
        selected_cells = hp.query_disc(
            nside, vector, radius, nest=True, fact=factor, inclusive=intersect
        )
        return np.isin(cell_ids, selected_cells, assume_unique=True)

    radius_ = buffer_size / sphere_radius

    masks = xr.apply_ufunc(
        _buffer_masks,
        cell_ids,
        positions,
        input_core_dims=[["cells"], ["cartesian"]],
        kwargs={
            "radius": radius_,
            "nside": nside,
            "factor": factor,
            "intersect": intersect,
        },
        output_core_dims=[["cells"]],
        vectorize=True,
    )

    return masks.assign_coords(cell_ids=cell_ids)


def create_masked_fill_map(tag, grid, maps, chunk_time=24, dims=["x", "y"]):
    """
    Create a masked fill map indicating the detection zones.

    Parameters:
    - tag (xarray.Dataset): A dataset containing station information, with variables 'recover_time' and 'deploy_time'.
    - grid (xarray.Dataset): A dataset containing grid information, with a variable 'time'.
    - maps (xarray.Dataset): A dataset containing map information, containing locations of acoustic stations.

    Returns:
    - fill_map (xarray.DataArray): A masked fill map indicating the detection zones.

    The function creates a masked fill map based on the station and grid information provided. It calculates
    the detection zones based on the time intervals specified in the 'recover_time' and 'deploy_time' variables
    in the tag station dataset. It then masks the fill map based on the grid's mask and returns the
    resulting fill map.
    """
    # load stations informations
    ds = tag["stations"].to_dataset()[["recover_time", "deploy_time"]]
    grid_time = grid[["time"]].cf.add_bounds(keys="time")

    # Expand dimensions of grid_time to match the number of deployment_ids
    stations = grid_time.expand_dims(deployment_id=ds.deployment_id, axis=0)

    # Create a boolean mask indicating if each time bin falls within
    # deploy_time and recover_time
    time_mask = (stations.time_bounds[:, 0] >= ds.deploy_time) & (
        stations.time_bounds[:, 1] <= ds.recover_time
    )

    # Add a new dimension 'time' to the dataset with the boolean mask
    stations["detecting"] = time_mask
    stations = stations.drop_vars(["time_bounds"])
    stations = stations.where(stations.sum(dim="time") != 0, other=True, drop=True)

    # Expand the maps dataset to match the dimensions of the active stations dataset
    all_detecting_stations = (
        maps.sel(deployment_id=stations.deployment_id)
        .expand_dims({"time": stations.time})
        .chunk({"time": chunk_time})
    )
    a = stations.sel(time=all_detecting_stations.time).chunk({"time": chunk_time})
    b = all_detecting_stations
    # keeps working with bool...
    ds = xr.ufuncs.logical_and(a, b).any(dim="deployment_id")

    all_detecting_stations = xr.where(ds == 0, 1, np.nan)
    # ...even though we still normalize (and thus turn the type to np.float64)
    fill_map = all_detecting_stations.detecting.pipe(utils.normalize, dim=dims)

    return fill_map


def emission_probability(
    tag,
    grid,
    buffer_size,
    nondetections="ignore",
    cell_ids="keep",
    chunk_time=24,
    dims=None,
):
    """construct emission probability maps from acoustic detections

    Parameters
    ----------
    tag : xarray.DataTree
        The tag data.
    grid : xarray.Dataset
        The target grid. Must have the ``cell_ids`` and ``time``
        coordinates and the ``mask`` variable.
    buffer_size : pint.Quantity
        The size of the buffer around each station. Must be given in
        a length unit.
    nondetections : {"mask", "ignore"}, default: "mask"
        How to deal with non-detections in time slices without detections:

        - "mask": set the buffer around stations without detections to `0`.
        - "ignore": all valid pixels are equally probable.
    cell_ids : {"recompute", "keep"}, default: "keep"
        How to deal with model cell ids for the computation of reception masks.

        - "keep": use the cell ids given by the model. This is the more correct method.
        - "recompute": recompute the cell ids based on the rotated lat / lon coords.
    dims : list of str, default: ["x", "y"]
        The spatial dimensions.

    Returns
    -------
    emission : xarray.Dataset
        The resulting emission probability maps.
    """
    if dims is None:
        dims = ["x", "y"]

    if "acoustic" not in tag or "stations" not in tag:
        return xr.Dataset()

    weights = (
        count_detections(
            tag["acoustic"].to_dataset(),
            by=grid[["time"]].cf.add_bounds(keys="time"),
        )
        .rename_vars({"count": "weights"})
        .chunk({"time": 1})
        .get("weights")
    )

    maps = deployment_reception_masks(
        tag["stations"].to_dataset(),
        grid[["cell_ids", "longitude", "latitude"]],
        buffer_size,
        method=cell_ids,
        dims=dims,
    ).chunk()  # chunks the map (to prevent autochunk which caused division by zero error)

    maps_index = maps.indexes["deployment_id"]
    weights_index = weights.indexes["deployment_id"]
    if weights_index.difference(maps_index, sort=False).size > 0:
        raise ValueError(
            "Some receiver ids in `tag.acoustic` are not included in `tag.stations`."
        )

    if nondetections == "ignore":
        fill_map = xr.ones_like(grid["cell_ids"], dtype=float).pipe(
            utils.normalize, dim=dims
        )
    elif nondetections == "mask":
        # fill_map = maps.any(dim="deployment_id").pipe(np.logical_not).astype(float)
        fill_map = create_masked_fill_map(tag, grid, maps, chunk_time, dims)
    else:
        raise ValueError("invalid nondetections treatment argument")

    return (
        maps.weighted(weights)
        .sum(dim="deployment_id")
        .transpose("time", *dims)
        .where((weights != 0).any(dim="deployment_id"), fill_map)
        .pipe(utils.normalize, dim=dims)
        .assign_attrs({"buffer_size": buffer_size.m_as("m")})
        .where(grid["mask"])
        .to_dataset(name="acoustic")
    )
