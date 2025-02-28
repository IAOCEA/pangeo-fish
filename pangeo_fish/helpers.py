import inspect
import os
import sys
import warnings
from pathlib import Path

import fsspec

# import hvplot.xarray
import holoviews as hv
import imageio as iio
import intake
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pint
import s3fs
import tqdm
import xarray as xr
import xdggs
from matplotlib.figure import Figure
from toolz.dicttoolz import valfilter
from toolz.functoolz import curry  # to change
from xarray_healpy import HealpyGridInfo, HealpyRegridder

import pangeo_fish.distributions as distrib
from pangeo_fish.acoustic import emission_probability
from pangeo_fish.cf import bounds_to_bins
from pangeo_fish.diff import diff_z
from pangeo_fish.grid import center_longitude
from pangeo_fish.hmm.estimator import EagerEstimator
from pangeo_fish.hmm.optimize import EagerBoundsSearch
from pangeo_fish.hmm.prediction import Gaussian1DHealpix, Gaussian2DCartesian
from pangeo_fish.io import (
    open_copernicus_catalog,
    open_tag,
    prepare_dataset,
    read_trajectories,
    save_html_hvplot,
    save_trajectories,
    tz_convert,
)
from pangeo_fish.pdf import combine_emission_pdf, normal
from pangeo_fish.tags import adapt_model_time, reshape_by_bins, to_time_slice
from pangeo_fish.utils import temporal_resolution
from pangeo_fish.visualization import filter_by_states, plot_map, render_frame

__all__ = [
    "to_healpix",
    "regrid_to_2d",
    "load_tag",
    "update_stations",
    "plot_tag",
    "load_model",
    "compute_diff",
    "open_diff_dataset",
    "regrid_dataset",
    "compute_emission_pdf",
    "compute_acoustic_pdf",
    "combine_pdfs",
    "optimize_pdf",
    "predict_positions",
    "plot_trajectories",
    "open_distributions",
    "plot_distributions",
    "render_frames",
    "render_distributions",
]


def _plot_in_figure(ds: xr.Dataset, **plot_kwargs) -> Figure:
    """Helper that plots statically `ds` in a plt.Figure."""
    fig, ax = plt.subplots()
    ds.plot(**(plot_kwargs | {"ax": ax}))
    return fig


def _save_zarr(ds: xr.Dataset, path: str, storage_options=None):
    """Helper for saving a .zarr array and warning in case of failure."""
    try:
        path_to_zarr = path + ".zarr" if not path.endswith(".zarr") else path
        ds.to_zarr(path_to_zarr, mode="w", storage_options=storage_options)
    except Exception:
        warnings.warn(
            f'An error occurred when saving a .zarr array to "{path_to_zarr}")',
            RuntimeWarning,
        )


def _inspect_curry_obj(curried_obj):
    """Inspect a curried object and retrieve its args and kwargs."""

    sig = inspect.signature(curried_obj.func)
    # default parameters
    params = {
        k: v.default if v.default is not inspect.Parameter.empty else None
        for k, v in sig.parameters.items()
    }

    # sig.parameters is ordered, so we can add the args of `curried_obj`
    for arg_name, arg_value in zip(sig.parameters.keys(), curried_obj.args):
        params[arg_name] = arg_value

    # finally, updates with the keywords (if any)
    params.update(curried_obj.keywords)
    return params


def _update_params_dict(factory, params: dict):
    """Inspect `factory` (assumed to be a curried object) to get its kw/args and update `params`.

    Note that `params` is updated with string representations of the arguments retrieved **except `cell_ids`**.

    Parameters
    -----------
    factory : Curried object
        It must have the attributes `func`, `args`, and `keywords`
    params : Dict
        The dictionary to update

    Returns
    --------
    params : Dict
        The updated dictionary
    """

    kwargs = {
        k: str(v) for k, v in _inspect_curry_obj(factory).items() if k != "cell_ids"
    }

    params["predictor_factory"] = {"class": str(factory), "kwargs": kwargs}

    return params


def to_healpix(ds: xr.Dataset) -> xr.Dataset:
    """Helper that loads a Dataset as a HEALPix grid (indexed by "cell_ids")."""

    ds["cell_ids"].attrs["grid_name"] = "healpix"
    attrs_to_keep = ["level", "grid_name"]
    ds["cell_ids"].attrs = {
        key: value
        for (key, value) in ds["cell_ids"].attrs.items()
        if key in attrs_to_keep
    }
    return ds.pipe(xdggs.decode)


def regrid_to_2d(ds: xr.Dataset):
    grid = HealpyGridInfo(level=ds.dggs.grid_info.level)
    return grid.to_2d(ds)


def load_tag(*, tag_root: str, tag_name: str, storage_options: dict = None, **kwargs):
    """Load a tag.

    Parameters
    ----------
    tag_root : str
        Path to the folder that contains the tag data under a folder `tag_name`
    tag_name : str
        Name of the tagged fish (e.g, "A19124"). Notably, It is used to fetch the biologging data from `{tag_root}/{tag_name}/`
    storage_options : dict, optional
        Dictionary containing storage options for connecting to the S3 bucket

    Returns
    -------
    tag : xr.DataTree
        The tag
    tag_log : xr.Dataset
        The DST data (sliced w.r.t released and recapture dates)
    time_slice : slice or like
        Time interval described by the released and recapture dates
    """

    tag = open_tag(tag_root, tag_name, storage_options)
    time_slice = to_time_slice(tag["tagging_events/time"])
    tag_log = tag["dst"].ds.sel(time=time_slice).assign_attrs({"tag_name": tag_name})
    return tag, tag_log, time_slice


def update_stations(
    *,
    tag: xr.DataTree,
    station_file_uri: str,
    method="merge",
    storage_options={},
    **kwargs,
):
    """Add or replace the acoustic receiver data of a tag.

    Parameters
    ----------
    tag : xr.DataTree
        The tag to update
    station_file_uri : str
        Path to the `.csv` file
    method : str, default: "merge"
        Operation to perform between the current and the new databases:
        - `merge` (default): the databases are merged.
        - `replace`: the current data is replaced by the new one.

    storage_options : dict, default: {}
        Dictionary containing storage options for connecting to the S3 bucket

    Returns
    -------
    tag : xr.DataTree
        The updated tag
    """

    if method not in ["merge", "replace"]:
        raise ValueError(f'Unknown method "{method}".')

    if not station_file_uri.endswith(".csv"):
        warnings.warn("`uri` should include the extension `.csv`.", UserWarning)
        station_file_uri += ".csv"

    with fsspec.open(station_file_uri, **storage_options) as file:
        stations = pd.read_csv(
            file,
            parse_dates=["deploy_time", "recover_time"],
            index_col="deployment_id",
        ).pipe(tz_convert, {"deploy_time": None, "recover_time": None})

        ds = xr.Dataset.from_dataframe(stations)
    if method == "merge":
        tag["stations"] = tag["stations"].ds.merge(ds)
    else:
        tag["stations"] = ds
    return tag


def plot_tag(
    *,
    tag: xr.DataTree,
    tag_log: xr.Dataset,
    save_html=False,
    target_root=".",
    storage_options: dict = None,
    **kwargs,
):
    """Plot a tag.

    Parameters
    ----------
    tag : xr.DataTree
        The tag
    tag_log : xr.Dataset
        The DST data
    save_html : bool, default: False
        Whether to save the plot as a HTML file
    target_root : str, default: "."
        Root of the folder to save `plot` as a HMTL file `tags.html`.
        Only used if `save_html=True`
    storage_options : dict, optional
        Dictionary containing storage options for connecting to the S3 bucket.
        Only used if `save_html=True` and that the saving is done on a S3 bucket.

    Returns
    -------
    hvplot : hv.Overlay
        Interactive plot of the pressure and temperature timeseries of the tag
    """

    plot = (
        (-tag["dst"].pressure).hvplot(width=1000, height=500, color="blue")
        * (-tag_log).hvplot.scatter(
            x="time", y="pressure", color="red", size=5, width=1000, height=500
        )
        * (
            (tag["dst"].temperature).hvplot(width=1000, height=500, color="blue")
            * (tag_log).hvplot.scatter(
                x="time", y="temperature", color="red", size=5, width=1000, height=500
            )
        )
    )  # type: hv.Overlay
    if save_html:
        try:
            path_to_html = Path(target_root) / "tags.html"
            # ensure the path is created (if needed) in case of local saving
            if storage_options is None:
                path_to_html.parent.mkdir(parents=True, exist_ok=True)
            save_html_hvplot(plot, str(path_to_html), storage_options=storage_options)
        except Exception as e:
            warnings.warn(
                "An error occurred when saving the Holoview plot of the tag:\n"
                + str(e),
                category=UserWarning,
            )
    return plot


def _open_copernicus_model(yaml_url: str, chunks: dict = None):
    """Open an intake catalog.

    Parameters
    ----------
    catalog_url : str
        Path to the `.yaml` file
    chunks : dict, optional
        How to chunk the data

    Returns
    -------
    model : xr.Dataset
        A dataset with the (notable) variables `TEMP`, `XE` and `H0`
    """
    cat = intake.open_catalog(yaml_url)
    model = open_copernicus_catalog(cat, chunks)
    return model


def _open_parquet_model(parquet_url: str):
    """Open a `.parq` dataset assembled with `virtualzarr`.

    Parameters
    ----------
    parquet_url : str
        Path to the `.parq` folders

    Returns
    -------
    model : xr.Dataset
        The dataset found
    """

    target_opts = {"anon": False}
    remote_opts = {"anon": False}
    fs = fsspec.filesystem(
        "reference",
        fo=parquet_url,
        remote_protocol="s3",
        remote_options=remote_opts,
        target_options=target_opts,
    )
    m = fs.get_mapper("")

    reference_ds = xr.open_dataset(
        m, engine="zarr", chunks={}, backend_kwargs={"consolidated": False}
    )
    reference_ds.coords["depth"].values[0] = 0.0
    return reference_ds


def load_model(
    *,
    uri: str,
    tag_log: xr.Dataset,
    time_slice: slice,
    bbox: dict[str, tuple[float, float]],
    chunk_time=24,
    **kwargs,
) -> xr.Dataset:
    """Load and prepare a reference model.

    Parameters
    ----------
    uri : str
        Path to the data. either an intake catalog (thus ending with `.yaml`) or a parquet array (thus ending with `.parq/`)
    tag_log : xr.Dataset
        The DST data
    time_slice : slice
        Time slice to sample the model from
    bbox : dict
        Spatial boundaries indexed by their coordinates (i.e, `longitude` and `latitude`) as well as the maximum depth indexed by `max_depth`.
    chunk_time : int, default: 24
        Chunk size for the time dimension

    Returns
    -------
    model : xr.Dataset
        The subset data
    """

    if uri.endswith(".yaml"):
        model = _open_copernicus_model(
            uri, chunks={"time": 8, "lat": -1, "lon": -1, "depth": -1}
        )
    elif uri.endswith(".parq") or uri.endswith(".parq/"):
        reference_ds = _open_parquet_model(uri)
        model = prepare_dataset(reference_ds)
    else:
        raise ValueError('Only intake catalogs and "parqued" data can be loaded.')

    reference_model = (
        model.sel(time=adapt_model_time(time_slice))
        .sel(lat=slice(*bbox["latitude"]), lon=slice(*bbox["longitude"]))
        .pipe(
            lambda ds: ds.sel(
                depth=slice(None, (bbox["max_depth"] - ds["XE"].min()).compute())
            )
        )
    ).chunk({"time": chunk_time, "lat": -1, "lon": -1, "depth": -1})
    return reference_model


def compute_diff(
    *,
    reference_model: xr.Dataset,
    tag_log: xr.Dataset,
    relative_depth_threshold: float,
    chunk_time=24,
    plot=False,
    save=False,
    target_root=".",
    storage_options: dict = None,
    **kwargs,
):
    """Compute the difference between the reference model and the DST data of a tag.
    Optionally, the dataset can be saved and plotted.

    Parameters
    ----------
    reference_model : xr.Dataset
        The reference model
    tag_log : xr.Dataset
        The DST data. *Hint: given a tag model, it corresponds to `tag["dst"].ds`*
    relative_depth_threshold : float
        Relative (seabed's) depth threshold to deal with cases where the fish's depths are lower than the seabed's depth
    chunk_time : int, default: 24
        Chunk size for the time dimension
    plot : bool, default: False
        Whether to return a plot of the dataset
    save : bool, default: False
        Whether to save the dataset
    target_root : str, default: "."
        Root of the folder to save the `.zarr` array (under `{target_root}/diff.zarr`)
        Only used if `save=True`
    storage_options : dict, optional
        Dictionary containing storage options for connecting to the S3 bucket.
        Only used if `save=True`

    Returns
    -------
    diff : xr.Dataset
        The difference between the biologging and field data
    figure : plt.Figure or None
        The plot of `diff`, or None if plot=False
    """

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
    ).chunk({"time": chunk_time})
    attrs = tag_log.attrs.copy()
    attrs.update({"relative_depth_threshold": relative_depth_threshold})
    diff = (
        diff_z(reference_model, reshaped_tag, depth_threshold=relative_depth_threshold)
        .assign_attrs(attrs)
        .assign(
            {
                "H0": reference_model["H0"],
                "ocean_mask": reference_model["H0"].notnull(),
            }
        )
    )  # type: xr.Dataset

    if save:
        _save_zarr(diff, f"{target_root}/diff.zarr", storage_options)

    figure = None
    if plot:
        try:
            diff = diff.compute()
            figure = _plot_in_figure(diff["diff"].count(["lat", "lon"]))
            [ax] = figure.get_axes()
            ax.set_title("Number of none-zero pixels")
        except Exception:
            warnings.warn(
                "An error occurred when plotting diff.",
                RuntimeWarning,
            )
    return diff, figure


def open_diff_dataset(*, target_root: str, storage_options: dict, **kwargs):
    """Ppen a diff dataset.

    Parameters
    ----------
    target_root : str
        Path root where to find `diff.zarr`
    storage_options : dict
        Additional information for `xarray` to open the `.zarr` array

    Returns
    -------
    ds : xr.Dataset
        The dataset
    """

    ds = (
        xr.open_dataset(
            f"{target_root}/diff.zarr",
            engine="zarr",
            chunks={},
            storage_options=storage_options,
        )
        .pipe(lambda ds: ds.merge(ds[["latitude", "longitude"]].compute()))
        .swap_dims({"lat": "yi", "lon": "xi"})
        .drop_vars(["lat", "lon"])
    )
    return ds


def regrid_dataset(
    *,
    ds: xr.Dataset,
    nside: int,
    min_vertices=1,
    rot={"lat": 0, "lon": 0},
    dims: list[str] = ["cells"],
    plot=False,
    save=False,
    target_root=".",
    storage_options: dict = None,
    **kwargs,
):
    """Regrid a dataset as a HEALPix grid, whose primary advantage is that all its cells/pixels cover the same surface area.

    Parameters
    ----------
    tag_log : xr.Dataset
        The DST data
    nside : int
        Tesolution of the HEALPix grid
    min_vertices : int, default: 1
        Minimum number of vertices for a valid transcription
    rot : Dict[str, Tuple[float, float]], default: {"lat": 0, "lon": 0}
        Mapping of angles to rotate the HEALPix grid. It must contain the keys "lon" and "lat"
    dims : List[str], default: ["cells"]
        The list of the dimensions for the regridding. Either ["x", "y"] or ["cells"]
    plot : bool, default: False
        Whether to return a plot of the dataset
    save : bool, default: False
        Whether to save the dataset
    target_root : str, default: "."
        Root of the folder to save the `.zarr` array (under `{target_root}/diff-regridded.zarr`).
        Only used if `save=True`
    storage_options : dict, optional
        Dictionary containing storage options for connecting to the S3 bucket.
        Only used if `save=True`

    Returns
    -------
    reshaped : xr.Dataset
        HEALPix version of `ds`
    figure : plt.Figure or None
        The plot of `reshaped`, or None if plot=False
    """

    grid = HealpyGridInfo(level=int(np.log2(nside)), rot=rot)
    target_grid = grid.target_grid(ds).pipe(center_longitude, 0)
    regridder = HealpyRegridder(
        ds[["longitude", "latitude", "ocean_mask"]],
        target_grid,
        method="bilinear",
        interpolation_kwargs={"mask": "ocean_mask", "min_vertices": min_vertices},
    )
    regridded = regridder.regrid_ds(ds)
    if dims == ["x", "y"]:
        reshaped = grid.to_2d(regridded).pipe(center_longitude, 0)
    elif dims == ["cells"]:
        reshaped = regridded.assign_coords(
            cell_ids=lambda ds: ds.cell_ids.astype("int64")
        )
    else:
        raise ValueError(f'Unknown dims "{dims}".')

    # adds the attributes found in `ds` as well as `min_vertices`
    attrs = ds.attrs.copy()
    attrs.update({"min_vertices": min_vertices})
    reshaped = reshaped.assign_attrs(attrs)

    if save:
        _save_zarr(reshaped, f"{target_root}/diff-regridded.zarr", storage_options)

    figure = False
    if plot:
        try:
            figure = _plot_in_figure(reshaped["diff"].count(dims))
            [ax] = figure.get_axes()
            ax.set_title("Number of none-zero pixels")
        except Exception:
            warnings.warn(
                "An error occurred when plotting the regridded dataset.",
                RuntimeWarning,
            )
    return reshaped, figure


def compute_emission_pdf(
    *,
    diff_ds: xr.Dataset,
    events_ds: xr.Dataset,
    differences_std: float,
    recapture_std: float,
    chunk_time: int = 24,
    dims: list[str] = ["cells"],
    plot=False,
    save=False,
    target_root=".",
    storage_options: dict = None,
    **kwargs,
):
    """Compute the temporal emission matrices given a dataset and tagging events.

    Parameters
    ----------
    diff_ds : xr.Dataset
        A dataset that must have the variables `diff` and `ocean_mask`
    events_ds : xr.Dataset
        The tagging events. It must have the coordinate `event_name` and values `release` and `fish_death`.
        *Hint: given a tag model, it corresponds to `tag["tagging_events"].ds`*
    differences_std : float
        Standard deviation that is applied to the data (passed to `scipy.stats.norm.pdf`). It'd express the estimated certainty of the field of difference
    recapture_std : float
        Covariance for the recapture event. It should reflect the certainty of the final recapture area
    dims : List[str], default: ["cells"]
        Spatial dimensions. Either ["x", "y"] or ["cells"]
    chunk_time : int, default: 24
        Chunk size for the time dimension
    plot : bool, default: False
        Whether to return a plot of the dataset
    save : bool, default: False
        Whether to save the dataset
    target_root : str, default: "."
        Root of the folder to save the `.zarr` array (under `{target_root}/emission.zarr`).
        Only used if `save=True`
    storage_options : dict, optional
        Dictionary containing storage options for connecting to the S3 bucket.
        Only used if `save=True`

    Returns
    -------
    emission_pdf : xr.Dataset
        The emission pdf
    figure : plt.Figure or None
        The plot of `emission_pdf`, or None if plot=False
    """

    if dims == ["x", "y"]:
        is_2d = True
    elif dims == ["cells"]:
        is_2d = False
    else:
        raise ValueError(f'Unknown dims "{dims}".')

    if not is_2d:
        diff_ds = to_healpix(diff_ds)

    grid = diff_ds[["latitude", "longitude"]].compute()
    initial_position = events_ds.sel(event_name="release")
    final_position = events_ds.sel(event_name="fish_death")

    if dims == ["x", "y"]:
        cov = distrib.create_covariances(1e-6, coord_names=["latitude", "longitude"])
        initial_probability = distrib.normal_at(
            grid,
            pos=initial_position,
            cov=cov,
            normalize=True,
            axes=["latitude", "longitude"],
        )
    else:
        initial_probability = distrib.healpix.normal_at(
            grid, pos=initial_position, sigma=1e-5
        )

    if final_position[["longitude", "latitude"]].to_dataarray().isnull().all():
        final_probability = None
    else:
        if is_2d:
            cov = distrib.create_covariances(
                recapture_std**2, coord_names=["latitude", "longitude"]
            )
            final_probability = distrib.normal_at(
                grid,
                pos=final_position,
                cov=cov,
                normalize=True,
                axes=["latitude", "longitude"],
            )
        else:
            final_probability = distrib.healpix.normal_at(
                grid, pos=final_position, sigma=recapture_std
            )

    emission_pdf = (
        normal(diff_ds["diff"], mean=0, std=differences_std, dims=dims)
        .to_dataset(name="pdf")
        .assign(
            valfilter(
                lambda x: x is not None,
                {
                    "initial": initial_probability,
                    "final": final_probability,
                    "mask": diff_ds["ocean_mask"],
                },
            )
        )
    )  # type: xr.Dataset
    attrs = diff_ds.attrs.copy()
    attrs.update({"differences_std": differences_std, "recapture_std": recapture_std})
    emission_pdf = emission_pdf.assign_attrs(attrs)
    emission_pdf = emission_pdf.chunk({"time": chunk_time} | {d: -1 for d in dims})

    if save:
        _save_zarr(emission_pdf, f"{target_root}/emission.zarr", storage_options)

    figure = False
    if plot:
        try:
            emission_pdf = emission_pdf.persist()
            figure = _plot_in_figure(emission_pdf["pdf"].count(dims))
            [ax] = figure.get_axes()
            ax.set_title("Number of none-zero pixels")
        except Exception:
            warnings.warn(
                "An error occurred when plotting the emission dataset.",
                RuntimeWarning,
            )
    return emission_pdf, figure


def compute_acoustic_pdf(
    *,
    emission_ds: xr.Dataset,
    tag: xr.DataTree,
    receiver_buffer: pint.Quantity,
    chunk_time=24,
    dims: list[str] = ["cells"],
    plot=False,
    save=False,
    target_root=".",
    storage_options: dict = None,
    **kwargs,
):
    """Compute a emission probability distribution from (acoustic) detection data.

    Parameters
    ----------
    emission_ds : xr.Dataset
        A dataset that must have the variables `time`, `mask` and `cell_ids`
    tag : xr.DataTree
        The tag data. It must have the datasets `acoustic` and `stations`
    receiver_buffer : pint.Quantity
        Maximum allowed detection distance for acoustic receivers
    chunk_time : int, default: 24
        Chunk size for the time dimension
    dims : List[str], default: ["cells"]
        The list of the dimensions. Either ["x", "y"] or ["cells"]
    plot : bool, default: False
        Whether to return a plot of the dataset
    save : bool, default: False
        Whether to save the dataset
    target_root : str, default: "."
        Root of the folder to save the `.zarr` array (under `{target_root}/acoustic.zarr`).
        Only used if `save=True`
    storage_options : dict, optional
        Dictionary containing storage options for connecting to the S3 bucket.
        Only used if `save=True`

    Returns
    -------
    acoustic_pdf : xr.Dataset
        The acoustic emission pdf
    figure : plt.Figure or None
        The plot of `acoustic_pdf`, or None if plot=False

    """

    if dims == ["cells"]:
        lon, lat = emission_ds["cell_ids"].attrs.get("lat", 0), emission_ds[
            "cell_ids"
        ].attrs.get("lon", 0)
        emission_ds = to_healpix(emission_ds)
        # adds back "lon" and "lat" keys
        emission_ds["cell_ids"].attrs["lon"] = lon
        emission_ds["cell_ids"].attrs["lat"] = lat

    acoustic_pdf = emission_probability(
        tag,
        emission_ds[["time", "cell_ids", "mask"]].compute(),
        receiver_buffer,
        nondetections="mask",
        cell_ids="keep",
        chunk_time=chunk_time,
        dims=dims,
    )
    attrs = emission_ds.attrs.copy()
    attrs.update({"receiver_buffer": str(receiver_buffer)})
    acoustic_pdf = acoustic_pdf.assign_attrs(attrs)

    if save:
        _save_zarr(acoustic_pdf, f"{target_root}/acoustic.zarr", storage_options)

    figure = False
    if plot:
        try:
            acoustic_pdf = acoustic_pdf.persist()
            figure = _plot_in_figure((acoustic_pdf["acoustic"] != 0).sum(dims))
            [ax] = figure.get_axes()
            ax.set_title("Sum of none-zero pixels")
        except Exception:
            warnings.warn(
                "An error occurred when plotting the acoustic dataset.",
                RuntimeWarning,
            )
    return acoustic_pdf, figure


def combine_pdfs(
    *,
    emission_ds: xr.Dataset,
    acoustic_ds: xr.Dataset,
    chunks: dict,
    dims=None,
    plot=False,
    **kwargs,
):
    """Combine and normalize 2 probability distributions (pdfs).

    Parameters
    ----------
    emission_ds : xr.Dataset
        Dataset of emission probabilities
    acoustic_ds : xr.Dataset
        Dataset of acoustic probabilities
    chunks : dict
        How to chunk the data
    dims : dict, optional
        Spatial dimensions to transpose the combined dataset. Relevant in case of a 2D, such as ["x", "y"] or ["y", "x"]
    plot : bool, default: False
        Whether to plot the sum of the distributions along the time dimension.


    Returns
    -------
    combined : xr.Dataset
        The combined pdf
    figure : plt.Figure, or None if `plot=False`
    """
    spatial_dims = [dim for dim in acoustic_ds.dims if dim != "time"]
    merged = emission_ds.merge(acoustic_ds)
    time_mask = xr.ufuncs.logical_or(
        (merged["acoustic"] == 0), xr.ufuncs.isnan(merged["pdf"])
    ).all(
        dim=spatial_dims
    )  # type: xr.DataArray

    num_times = time_mask.sum().to_numpy().item()  # type: int
    if num_times != 0:
        warnings.warn(f"The combined pdf sums to 0 for {num_times} times.", UserWarning)
        # temparory fix: replaces by the values from acoustic
        time_mask = time_mask.compute()
        mask = (merged["mask"].notnull()).compute()
        # TODO: is this actually correct? Even though we normalize, isn't the prod of the distributions half of `acoustic_ds["acoustic"]`?
        # careful here: we erase "pdf" by "acoustic" with `acoustic`: using acoustic_ds may erase attributes!
        merged["pdf"][time_mask] = acoustic_ds["acoustic"][time_mask].where(
            mask, drop=False
        )

    combined = merged.pipe(combine_emission_pdf).chunk(chunks)

    # optional spatial transposition
    if (dims is not None) and ("cells" not in dims):
        # TODO: still not enough...
        if not all([d in combined.dims for d in dims]):
            raise ValueError(
                f'Not all the dimensions provided (dims="{dims}") were found in the emission distribution.'
            )
        if "time" in dims:
            warnings.warn(
                '"time" was found in "dims". Spatial dimensions are expected.',
                UserWarning,
            )
            combined = combined.transpose(*dims)
        else:
            combined = combined.transpose("time", *dims)

    # last operation: ensure no attributes are lost during the combination
    for ds in [acoustic_ds, emission_ds]:
        combined.attrs.update(ds.attrs)
        if combined.coords.get("cell_ids", None) is not None:
            combined["cell_ids"].attrs.update(ds["cell_ids"].attrs)

    figure = False
    if plot:
        try:
            figure = _plot_in_figure(combined["pdf"].sum(dims), ylim=(0, 2))
            [ax] = figure.get_axes()
            ax.set_title("Sum of the probabilities")
        except Exception:
            warnings.warn(
                "An error occurred when plotting the combined dataset.",
                RuntimeWarning,
            )
    return combined, figure


def _get_predictor_factory(ds: xr.Dataset, truncate: float, dims: list[str]):
    if dims == ["x", "y"]:
        predictor = curry(Gaussian2DCartesian, truncate=truncate)
    elif dims == ["cells"]:
        predictor = curry(
            Gaussian1DHealpix,
            cell_ids=ds["cell_ids"].data,
            grid_info=ds.dggs.grid_info,
            truncate=truncate,
            weights_threshold=1e-8,
            pad_kwargs={"mode": "constant", "constant_value": 0},
            optimize_convolution=True,
        )
    else:
        raise ValueError(f'Unknown dims "{dims}".')
    return predictor


def _get_max_sigma(
    ds: xr.Dataset,
    earth_radius: pint.Quantity,
    adjustment_factor: float,
    truncate: float,
    maximum_speed: pint.Quantity,
    as_radians: bool,
) -> float:
    earth_radius_ = xr.DataArray(earth_radius, dims=None)

    timedelta = temporal_resolution(ds["time"]).pint.quantify().pint.to("h")
    grid_resolution = earth_radius_ * ds["resolution"].pint.quantify()

    maximum_speed_ = xr.DataArray(maximum_speed, dims=None).pint.to("km / h")
    if as_radians:
        max_grid_displacement = (
            maximum_speed_ * timedelta * adjustment_factor / earth_radius_
        )
    else:  # in pixels
        max_grid_displacement = (
            maximum_speed_ * timedelta * adjustment_factor / grid_resolution
        )
    max_sigma = max_grid_displacement.pint.to("dimensionless").pint.magnitude / truncate

    return max_sigma.item()


def optimize_pdf(
    *,
    ds: xr.Dataset,
    earth_radius: pint.Quantity,
    adjustment_factor: float,
    truncate: float,
    maximum_speed: pint.Quantity,
    tolerance: float,
    dims: list[str] = ["cells"],
    save_parameters=False,
    storage_options: dict = None,
    target_root=".",
    **kwargs,
) -> dict:
    """Optimize a temporal probability distribution.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset of emission probabilities
    earth_radius : pint.Quantity
        Radius of the Earth used for distance calculations
    adjustment_factor : float
        Factor value for the maximum fish's displacement
    truncate : float
        Truncating factor for convolution process
    maximum_speed : pint.Quantity
        Maximum fish's velocity
    tolerance : float
        Tolerance level for the optimised parameter search computation
    dims : List[str], default: ["cells"]
        The list of the dimensions. Either ["x", "y"] or ["cells"]
    save_parameters : bool, default: False
        Whether to save the results under `{target_root}/parameters.json`
    target_root : str, default: "."
        Root of the folder to save the results as a json file `parameters.json`
        Only used if `save_parameters=True`
    storage_options : dict, optional
        Dictionary containing storage options for connecting to the S3 bucket.
        Only used if `save_parameters=True` and that the saving is done on a S3 bucket.

    Returns
    -------
    params : dict
        A dictionary containing the optimization results (mainly, the sigma value of the Brownian movement model)
    """

    # it is important to compute before re-indexing? Yes.
    ds = ds.compute()

    if "cells" in ds.dims:
        ds = to_healpix(ds)
        as_radians = True
    else:
        as_radians = False

    max_sigma = _get_max_sigma(
        ds, earth_radius, adjustment_factor, truncate, maximum_speed, as_radians
    )
    predictor_factory = _get_predictor_factory(ds=ds, truncate=truncate, dims=dims)

    estimator = EagerEstimator(sigma=None, predictor_factory=predictor_factory)
    ds.attrs["max_sigma"] = max_sigma  # limitation of the helper

    optimizer = EagerBoundsSearch(
        estimator,
        (1e-4, ds.attrs["max_sigma"]),
        optimizer_kwargs={"disp": 3, "xtol": tolerance},
    )
    optimized = optimizer.fit(ds)
    params = optimized.to_dict()  # type: dict
    _update_params_dict(factory=predictor_factory, params=params)

    if save_parameters:
        try:
            path_to_json = Path(target_root) / "parameters.json"
            if storage_options is None:
                path_to_json.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame.from_dict(params, orient="index").to_json(
                str(path_to_json), storage_options=storage_options
            )
        except Exception:
            warnings.warn(
                f'An error occurred when attempting to export the results under "{path_to_json}".',
                RuntimeWarning,
            )
    return params


def predict_positions(
    *,
    target_root: str,
    storage_options: dict,
    chunks: dict,
    track_modes=["mean", "mode"],
    additional_track_quantities=["speed", "distance"],
    save=True,
    **kwargs,
):
    """High-level helper function for predicting fish's positions and generating the consequent trajectories.
    It futhermore saves the latter under `states.zarr` and `trajectories.parq`.

    .. warning::
        `target_root` must not end with "/".

    Parameters
    ----------
    target_root : str
        Path to a folder that must contain a folder `combined.zarr` and the file `parameters.json`
    storage_options : dict
        Additional information for `xarray` to open the `.zarr` array
    chunks : dict
        Chunk size to load the xr.Dataset `combined.zarr`
    track_modes : list[str], default: ["mean", "mode"]
        Options for decoding trajectories. See `pangeo_fish.hmm.estimator.EagerEstimator.decode`
    additional_track_quantities : list[str], default: ["speed", "distance"]
        Additional quantities to compute from the decoded tracks. See `pangeo_fish.hmm.estimator.EagerEstimator.decode`
    save : bool, default: True
        Whether to save the `states` distribution and the trajectories.

    Returns
    -------
    states : xr.Dataset
        A geolocation model, i.e., positional temporal probabilities
    trajectories : TrajectoryCollection
        The tracks decoded from `states`
    """

    # loads the normalized .zarr array
    emission = xr.open_dataset(
        f"{target_root}/combined.zarr",
        engine="zarr",
        chunks=chunks,
        inline_array=True,
        storage_options=storage_options,
    )
    emission = emission.compute()

    if "cells" in emission.dims:
        emission = to_healpix(emission)

    params = pd.read_json(
        f"{target_root}/parameters.json", storage_options=storage_options
    ).to_dict()[0]

    # do not account for the other kwargs...
    # not very robust yet...
    truncate = float(params["predictor_factory"]["kwargs"]["truncate"])
    cls_name = params["predictor_factory"]["class"]  # type: str
    if "Gaussian2DCartesian" in cls_name:
        predictor_factory = _get_predictor_factory(
            emission, truncate=truncate, dims=["x", "y"]
        )
    elif "Gaussian1DHealpix" in cls_name:
        predictor_factory = _get_predictor_factory(
            emission, truncate=truncate, dims=["cells"]
        )
    else:
        raise RuntimeError("Could not infer predictor's class from the `.json` file.")

    optimized = EagerEstimator(
        sigma=params["sigma"], predictor_factory=predictor_factory
    )

    states = optimized.predict_proba(emission)
    states = states.to_dataset().chunk(chunks)  # type: xr.Dataset
    states.attrs.update(emission.attrs)

    if save:
        _save_zarr(states, f"{target_root}/states.zarr", storage_options)

    trajectories = optimized.decode(
        emission,
        states.fillna(0),
        mode=track_modes,
        progress=False,
        additional_quantities=additional_track_quantities,
    )

    if save:
        save_trajectories(trajectories, target_root, storage_options, format="parquet")

    return states, trajectories


def plot_trajectories(
    *,
    target_root: str,
    track_modes: list[str],
    storage_options: dict,
    save_html=True,
    **kwargs,
):
    """Read decoded trajectories and plots an interactive visualization.
    Optionally, the plot can be saved as a HTML file.

    .. warning::
        `target_root` must not end with "/".

    Parameters
    ----------
    target_root : str
        Path to a folder that must contain a trajectory collection `trajectories.parq`
    track_modes : list[str]
        Names of the tracks
    storage_options : dict
        Additional information for `xarray` to open the `.zarr` array
    save_html : bool, default: True
        Whether to save the plot (under `{target_root}/trajectories.html`)

    Returns
    -------
    plot : hv.Layout
        Interactive plot of the trajectories
    """

    trajectories = read_trajectories(
        track_modes, target_root, storage_options, format="parquet"
    )
    plots = [
        traj.hvplot(
            c="speed",
            tiles="CartoLight",
            title=traj.id,
            cmap="cmo.speed",
            width=300,
            height=300,
        )
        for traj in trajectories.trajectories
    ]
    plot = hv.Layout(plots).cols(2)

    if save_html:
        path_to_html = Path(target_root) / "trajectories.html"
        if storage_options is None:
            path_to_html.parent.mkdir(parents=True, exist_ok=True)
        save_html_hvplot(plot, str(path_to_html), storage_options)

    return plot


def open_distributions(
    *, target_root: str, storage_options: dict, chunks: dict, chunk_time=24, **kwargs
):
    """Load and merge the `emission` and `states` probability distributions into a single dataset.

    .. warning::
        Since this function is assumed to be used for visualization and rendering tasks,\
        and that only 2D-indexed data is currently supported by `pangeo-fish`, **the dataset returned is regridded to 2D.**

    Parameters
    ----------
    target_root : str
        Path to a folder that must contain the `combined.zarr` and `states.zarr` files.
        **Must not end with "/".**
    storage_options : dict
        Additional information for `xarray` to open the `.zarr` array
    chunks : dict
        Mapping of the chunk sizes for each dimension of the xr.Datasets to load: namely, the `.zarr` arrays `combined` and `states`
    chunk_time : int, default: 24
        Chunk size of the dimension "time" to use to chunk the result

    Returns
    -------
    data : xr.Dataset
        The merged and cleaned dataset, 2D-indexed

    See Also
    --------
    `helpers.plot_distributions()` and `helpers.render_distributions()`.
    """

    emission = (
        xr.open_dataset(
            f"{target_root}/combined.zarr",
            engine="zarr",
            chunks=chunks,
            inline_array=True,
            storage_options=storage_options,
        )
        .rename_vars({"pdf": "emission"})
        .drop_vars(["final", "initial"], errors="ignore")
    )
    states = xr.open_dataset(
        f"{target_root}/states.zarr",
        engine="zarr",
        chunks=chunks,
        inline_array=True,
        storage_options=storage_options,
    ).where(emission["mask"])

    data = xr.merge([states, emission.drop_vars(["mask"])])

    # if the data is 1D indexed, regrid it to 2D
    # since this function is expected to be used for plotting and rendering tasks
    if "cells" in data.dims:
        data = to_healpix(data)
        data = regrid_to_2d(data)

    data = data.assign_coords(longitude=((data["longitude"] + 180) % 360 - 180))
    data = data.chunk({d: -1 if d != "time" else chunk_time for d in data.dims})

    return data


def plot_distributions(*, data: xr.Dataset, bbox=None, **kwargs):
    """Plot an interactive visualization of dataset resulting from the merging of `emission` and the `states` distributions.
    See `pangeo_fish.helpers.open_distributions()`.

    Parameters
    ----------
    data : xr.Dataset
        A dataset that contains the `emission` and `states` variables
    bbox : dict[str, tuple[float, float]], optional
        The spatial boundaries of the area of interest. Must have the keys "longitude" and "latitude".

    Returns
    -------
    plot : hv.Layout
        Interactive plot of the `states` and `emission` distributions
    """

    # TODO: adding coastlines reverts the xlim / ylim arguments
    plot1 = plot_map(data["states"], bbox)
    plot2 = plot_map(data["emission"], bbox)
    plot = hv.Layout([plot1, plot2]).cols(2)

    return plot


def render_frames(*, ds: xr.Dataset, time_slice: slice = None, **kwargs):
    """Helper function for rendering images.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset which the `emission` and `states` variables
    time_slice : slice, default: None
        Timesteps to render. If not provided, all timesteps are rendered (not recommended)

    Returns
    -------
    None : None
        Nothing is returned

    Other Parameters
    ---------------
    kwargs : dict
        Additional arguments passed to `pangeo-fish.visualization.render_frame()`.
        See its documentation for more information.
    """

    if time_slice is not None:
        ds = ds.isel(time=time_slice)

    ds.map_blocks(
        render_frame, kwargs=kwargs, template=ds
    ).compute()  # to trigger the computation


def _render_video(frames_fp: list[str], video_fn: str, extension="gif", fps=10) -> str:

    def _is_format_available(format_name: str):
        formats = iio.config.known_plugins.keys()
        return format_name in formats

    if extension == "gif":
        kwargs = dict(uri=f"{video_fn}.gif", mode="I", fps=fps)

    elif extension == "mp4":
        if not _is_format_available("FFMPEG"):
            raise ModuleNotFoundError(
                "FFMPEG not found: have you installed imageio[ffmpeg]?"
            )

        kwargs = dict(
            uri=f"{video_fn}.mp4",
            mode="I",
            fps=fps,
            format="FFMPEG",
            codec="libx264",
            pixelformat="yuv420p",
            ffmpeg_params=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
        )

    else:
        raise ValueError(f'Unknown extension "{extension}".')

    pbar = tqdm.tqdm(sorted(frames_fp), file=sys.stdout)
    pbar.set_description("Rendering the video...")
    with iio.get_writer(**kwargs) as writer:
        for filename in pbar:
            image = iio.v3.imread(filename)
            writer.append_data(image)
    pbar.close()
    return kwargs["uri"]


def render_distributions(
    *,
    data: xr.Dataset,
    time_step=3,
    frames_dir="frames",
    output_path="states",
    extension="gif",
    fps=10,
    remove_frames=True,
    storage_options: dict = None,
    **kwargs,
):
    """Render a video of a dataset resulting from the merging of `emission` and the `states` distributions.
    See `pangeo_fish.helpers.open_distributions()`.

    Parameters
    ----------
    data : xr.Dataset
        A dataset that contains the `emission` and `states` variables
    time_step : int, default: 3
        Time step to sample data from `data`
    frames_dir : str, default: "frames"
        Name of the folder to save the images to
    output_path : str, default: "states"
        Path to save the video. In case of an AWS S3 uri, the video is first saved locally, and then send to the bucket
    extension : str, default: "gif"
        Name of the file extension of the video
        Either "gif" or "mp4". **In the latter case, make sure to have installed imageio[ffmpeg]**
    fps : int, default: 10
        Number of frames per second.
    remove_frames : bool, default: True
        Whether to delete the frames
    storage_options : dict, optional
        Dictionary containing storage options for connecting to the S3 bucket


    Returns
    -------
    video_fn : str
        Local path to the video
    """

    # os.path.split(.) removes the "/"!
    path_root, filename = os.path.split(output_path)
    filename = filename.split(".")[0]

    # quick input checking
    if not all(var_name in data.variables for var_name in ["emission", "states"]):
        raise ValueError(
            '"emission" and/or "states" variable(s) not found in the dataset.'
        )
    if sorted(list(data.dims)) != ["time", "x", "y"]:
        raise ValueError(
            'The dataset must have its dimensions equal to ["time", "x", "y"].'
        )

    time_slice = slice(0, data["time"].size - 1, time_step)
    sliced_data = (
        data.isel(time=time_slice)
        .chunk({"time": 1, "x": -1, "y": -1})
        .pipe(lambda ds: ds.merge(ds[["longitude", "latitude"]].compute()))
    ).pipe(
        filter_by_states
    )  # type: xr.Dataset
    # add a time index
    sliced_data = sliced_data.assign_coords(
        time_index=("time", np.arange(sliced_data.sizes["time"]))
    )
    sliced_data = sliced_data.chunk({"time": 1, "x": -1, "y": -1})

    path_to_frames = Path(frames_dir)
    path_to_frames.mkdir(parents=True, exist_ok=True)

    # see pangeo-fish.visualization.render_frame()
    render_frames(ds=sliced_data, **(kwargs | {"frames_dir": frames_dir}))
    try:
        video_fp = _render_video(
            frames_fp=[file.resolve() for file in path_to_frames.glob("*.png")],
            video_fn=filename,
            extension=extension,
            fps=fps,
        )
        if path_root.startswith("s3://"):
            if storage_options is None:
                warnings.warn(
                    "Remote video uploading to S3 cancelled: no storage options found.",
                    RuntimeWarning,
                )
            else:
                s3 = s3fs.S3FileSystem(**storage_options)
                s3.put_file(video_fp, f"{path_root}/{video_fp}")
    except Exception as e:
        warnings.warn(
            "An error occurred when rendering the video:\n" + str(e), RuntimeWarning
        )
        video_fp = ""
    finally:
        pbar = tqdm.tqdm(path_to_frames.glob("*.png"), file=sys.stdout)
        pbar.set_description("Removing .png files")
        # we only know that the images are stored in `path_to_frames`
        if remove_frames:
            for filepath in pbar:
                os.remove(filepath)
        pbar.close()
    return video_fp
