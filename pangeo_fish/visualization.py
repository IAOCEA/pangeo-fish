"""Rendering functions."""

import warnings

import cartopy.crs as ccrs
import cartopy.feature as cf
import cmocean  # noqa: F401
import hvplot.xarray  # noqa: F401
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from shapely.errors import ShapelyDeprecationWarning


def filter_by_states(ds):
    return ds.where(ds["states"].sum(dim="time", skipna=True).compute() > 0, drop=True)


def create_single_frame(ds: xr.Dataset, figure, **kwargs):
    """Default function for plotting a snapshot (i.e, **timeless** data) of the `emission` and `states` distributions.

    Parameters
    ----------
    ds : xr.Dataset
        A **timeless** dataset, i.e., whose dimensions are `[x, y]`, that has the `emission` and `states` variables.
    figure : A matplotlib Figure
        The figure to which add the axes and plots
    xlim : tuple[float, float], optional
        The longitude interval to plot
    ylim : tuple[float, float], optional
        The latitude interval to plot
    vmax : dict[str, float], optional
        Mapping of the maximum values for coloring the plots, indexed by "emission" and "states"

    Returns
    -------
    None : None
        Nothing is returned
    """

    if sorted(list(ds.dims)) != ["x", "y"]:
        raise ValueError(
            f"Malformed dataset (dims of {list(ds.dims)} instead of [x, y])."
        )

    warnings.filterwarnings(
        action="ignore",
        category=ShapelyDeprecationWarning,  # in cartopy
    )
    warnings.filterwarnings(
        action="ignore",
        category=UserWarning,
        message=r"No `(vmin|vmax)` provided. Data limits are calculated from input. Depending on the input this can take long. Pass `\1` to avoid this step",
    )

    ds_ = ds.drop_vars(["resolution"], errors="ignore")

    projection = ccrs.Mercator()
    crs = ccrs.PlateCarree()

    default_xlim = [ds_["longitude"].min(), ds_["longitude"].max()]
    default_ylim = [ds_["latitude"].min(), ds_["latitude"].max()]
    default_vmax = {
        "states": ds_["states"].max().to_numpy().item(),
        "emission": ds_["emission"].max().to_numpy().item(),
    }

    x0, x1 = kwargs.get("xlim", default_xlim)
    y0, y1 = kwargs.get("ylim", default_ylim)
    default_vmax.update(kwargs.get("vmax", {}))

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    cbar_kwargs = {
        "orientation": "horizontal",
        "shrink": 0.65,
        "pad": 0.05,
        "aspect": 50,
        "format": formatter,
        "use_gridspec": True,
        "extend": "max",
    }
    gridlines_kwargs = {
        "crs": crs,
        "draw_labels": True,
        "linewidth": 0.6,
        "color": "gray",
        "alpha": 0.5,
        "linestyle": "-.",
    }
    plot_kwargs = {
        "x": "longitude",
        "y": "latitude",
        "transform": ccrs.PlateCarree(),
        "cmap": "cool",
        "xlim": [x0, x1],
        "ylim": [y0, y1],
        "vmin": 0,
    }
    gs = figure.add_gridspec(nrows=1, ncols=2, hspace=0, wspace=-0.2, top=0.925)
    (ax1, ax2) = gs.subplots(
        subplot_kw={"projection": projection, "frameon": True},
        sharex=True,
        sharey=True,
    )

    ds_["states"].plot(
        **(
            plot_kwargs
            | {
                "ax": ax1,
                "cbar_kwargs": cbar_kwargs | {"label": "State Probability"},
                "vmax": default_vmax["states"],
            }
        )
    )
    ax1.set_title("")
    ax1.add_feature(cf.COASTLINE.with_scale("10m"), lw=0.5)
    ax1.add_feature(cf.BORDERS.with_scale("10m"), lw=0.3)
    ax1.set_extent([x0, x1, y0, y1], crs=crs)
    gl1 = ax1.gridlines(**gridlines_kwargs)
    gl1.right_labels = False
    gl1.top_labels = False

    ds_["emission"].plot(
        **(
            plot_kwargs
            | {
                "ax": ax2,
                "cbar_kwargs": cbar_kwargs | {"label": "Emission Probability"},
                "vmax": default_vmax["emission"],
            }
        )
    )
    ax2.set_title("")
    ax2.add_feature(cf.COASTLINE.with_scale("10m"), lw=0.5)
    ax2.add_feature(cf.BORDERS.with_scale("10m"), lw=0.3)
    ax2.set_extent([x0, x1, y0, y1], crs=crs)

    gl2 = ax2.gridlines(**gridlines_kwargs)
    gl2.left_labels = False
    gl2.top_labels = False

    return None


def render_frame(ds: xr.Dataset, *args, figsize=(14, 8), frames_dir=".", **kwargs):
    """
    .. warning::
        Designed to be used with `dask.map_blocks()`.
        As such, `ds` must have the following variables:
        - `time_index`, representing the time index. It is used for naming the image (`.png`)
        - `emission` and `states`, the data to plot


    Used along with `dask.map_blocks()`, it will call create_single_frame() for each timestep,\
    and save the consequent images under `{frames_dir}/frame_XXXXX.png`.

    Parameters
    ----------
    frames_dir : str, default: "."
        Name of the folder to save the frame
    figsize : tuple[float, float], default: (14, 8)
        Name of the folder to save the frame

    Returns
    -------
    ds : xr.Dataset
        The input dataset (see `dask.map_blocks()`)
    """

    figure = plt.figure(figsize=figsize)  # figsize=(12, 6)

    try:
        if ds.sizes["time"] > 1:
            warnings.warn(
                f"Multiple timesteps detected in `ds` (size: {ds.sizes['time']}): only the first one will be rendered.",
                UserWarning,
            )

        create_single_frame(ds.isel(time=0), figure, **kwargs)  # xr.Dataset.squeeze()?
        time = ds["time"].values[0]
        title = f"Time = {np.datetime_as_string(time, unit='s')}"
        figure.suptitle(title)

        time_index = ds["time_index"].values[0]
        figure.savefig(
            f"{frames_dir}/frame_{time_index:05d}.png"
        )  # , bbox_inches="tight", pad_inches=0.2)
    except Exception as e:
        print(
            f"============ Exception at time {ds['time_index'].values[0]} =============="
        )
        print(e)
        print("=========================================================")
    finally:
        plt.close(figure)

    return ds


def plot_map(
    arr,
    bbox,
    x="longitude",
    y="latitude",
    rasterize=True,
    geo=True,
    coastline="10m",
    tiles=None,
    cmap="cmo.amp",
    **kwargs,
):
    """wrapper around `DataArray.hvplot.quadmesh`, with different defaults"""
    return arr.hvplot.quadmesh(
        x=x,
        y=y,
        xlim=bbox["longitude"],
        ylim=bbox["latitude"],
        rasterize=rasterize,
        geo=geo,
        coastline=coastline,
        tiles=tiles,
        cmap=cmap,
        **kwargs,
    )


def plot_trajectories(trajectories, *, subplots=False, **kwargs):
    import holoviews as hv

    if not subplots:
        return trajectories.hvplot(**kwargs)
    else:
        plots = [traj.hvplot(title=traj.id, **kwargs) for traj in trajectories]
        return hv.Layout(plots).cols(2)
