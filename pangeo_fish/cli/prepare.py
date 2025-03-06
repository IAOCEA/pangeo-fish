import numpy as np
import xarray as xr
from xarray_healpy import HealpyGridInfo, HealpyRegridder

from pangeo_fish.cf import bounds_to_bins
from pangeo_fish.diff import diff_z
from pangeo_fish.distributions import create_covariances, normal_at
from pangeo_fish.grid import center_longitude
from pangeo_fish.pdf import normal
from pangeo_fish.tags import adapt_model_time, reshape_by_bins, to_time_slice
from pangeo_fish.utils import temporal_resolution


def subtract_data(tag, model, parameters):
    # prepare tag
    time_slice = to_time_slice(tag["tagging_events/time"])
    tag_log = tag["dst"].ds.sel(time=time_slice)

    bbox = parameters["bbox"]

    reference_model = (
        # align model time with tag log
        model.sel(time=adapt_model_time(time_slice))
        # subset the data to the region of interest using a bbox
        .sel(lat=slice(*bbox["lat"]), lon=slice(*bbox["lon"]))
        # drop data for depth layers that are too unlikely
        .pipe(
            lambda ds: ds.sel(
                depth=slice(
                    None, (tag_log["pressure"].max() - ds["XE"].min()).compute()
                )
            )
        )
    )

    # subtract tag data from the model
    reshaped_tag = reshape_by_bins(
        tag_log,
        dim="time",
        bins=(
            reference_model.cf.add_bounds(["time"], output_dim="bounds")
            .pipe(bounds_to_bins, bounds_dim="bounds")
            .get("time_bins")
        ),
        bin_dim="bincount",
        other_dim="obs",
    ).chunk({"time": 1})

    diff = diff_z(
        reference_model.chunk({"depth": -1}),
        reshaped_tag,
        depth_threshold=parameters["relative_depth_threshold"],
    ).assign({"H0": reference_model["H0"], "mask": reference_model["H0"].notnull()})

    return diff


def regrid(ds, parameters):
    grid = HealpyGridInfo(
        level=int(np.log2(parameters["nside"])), rot=parameters["rot"]
    )
    target_grid = grid.target_grid(ds).pipe(center_longitude)

    regridder = HealpyRegridder(
        ds[["longitude", "latitude", "mask"]],
        target_grid,
        method="bilinear",
        interpolation_kwargs={
            "mask": "mask",
            "min_vertices": parameters["min_vertices"],
        },
    )
    regridded = regridder.regrid_ds(ds)
    reshaped = grid.to_2d(regridded).pipe(center_longitude)

    return reshaped


def position_pdf(grid, position, sigma):
    if not position:
        return None

    coords = ["longitude", "latitude"]
    cov = create_covariances(sigma, coord_names=coords)

    return normal_at(grid, pos=position, cov=cov, normalize=True, axes=coords)


def maximum_parameter(ds, parameters):
    earth_radius = xr.DataArray(parameters["earth_radius"], dims=None)

    timedelta = temporal_resolution(ds["time"]).pint.quantify().pint.to("h")
    grid_resolution = earth_radius * ds["resolution"].pint.quantify()

    maximum_speed = xr.DataArray(parameters["maximum_speed"], dims=None).pint.to(
        "km / h"
    )
    maximum_grid_displacement = (
        maximum_speed * timedelta * parameters["adjustment_factor"] / grid_resolution
    )

    max_sigma = (
        maximum_grid_displacement.pint.to("dimensionless").pint.magnitude
        / parameters["truncate"]
    )

    return max_sigma


def temperature_emission_matrices(ds, tag, parameters):
    grid = ds[["longitude", "latitude"]].compute()

    additional_variables = {
        "initial": position_pdf(
            grid,
            tag["tagging_events"].ds.sel(event_name="release"),
            sigma=parameters["release_std"],
        ),
        "final": position_pdf(
            grid,
            tag["tagging_events"].ds.sel(event_name="fish_death"),
            sigma=parameters["recapture_std"],
        ),
        "mask": ds["mask"],
    }
    if additional_variables["final"] is None:
        del additional_variables["final"]

    return (
        normal(ds["diff"], mean=0, std=parameters["differences_std"], dims=["y", "x"])
        .to_dataset(name="temperature")
        .assign(additional_variables)
        .assign_attrs(ds.attrs | {"max_sigma": maximum_parameter(ds, parameters)})
    )
