import json
from contextlib import nullcontext
from functools import partial

import pint_xarray
import rich_click as click
import xarray as xr
from rich.console import Console

from pangeo_fish.cli.cluster import create_cluster
from pangeo_fish.cli.path import construct_target_root
from pangeo_fish.hmm.estimator import CachedEstimator, EagerEstimator
from pangeo_fish.pdf import combine_emission_pdf

ureg = pint_xarray.unit_registry

click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True
console = Console()


def decode_parameters(obj):
    if list(obj) != ["magnitude", "units"]:
        return obj

    return ureg.Quantity(obj["magnitude"], obj["units"])


def maybe_compute(ds, compute):
    if not compute:
        return ds

    return ds.compute()


def create_cached_estimator(cache_path, **kwargs):
    import zarr

    cache_store = zarr.storage.DirectoryStore(cache_path)

    return CachedEstimator(cache=cache_store, **kwargs)


@click.group()
def main():
    """Run the pangeo-fish model."""
    pass


@main.command(
    "prepare",
    short_help="transform the data into something suitable for the model to run",
)
@click.option("--cluster-definition", type=click.File(mode="r"))
@click.argument("parameters", type=click.File(mode="r"))
@click.argument("runtime_config", type=click.File(mode="r"))
def prepare(parameters, runtime_config, cluster_definition):
    """Transform the input data into a set of emission parameters"""
    import intake

    from pangeo_fish import acoustic
    from pangeo_fish.cli.prepare import (
        regrid,
        subtract_data,
        temperature_emission_matrices,
    )
    from pangeo_fish.io import open_copernicus_catalog, open_tag

    runtime_config = json.load(runtime_config)
    parameters = json.load(parameters, object_hook=decode_parameters)
    cluster_definition = json.load(cluster_definition)

    target_root = construct_target_root(runtime_config, parameters)
    target_root.mkdir(parents=True, exist_ok=True)

    with (
        create_cluster(**cluster_definition) as client,
        console.status("[bold blue]processing[/]") as status,
    ):
        console.print(f"[bold white]dashboard link[/]: {client.dashboard_link}")

        # open tag
        tag = open_tag(runtime_config["tag_root"], parameters["tag_name"])
        console.log("successfully opened tag log")

        # open model
        cat = intake.open_catalog(runtime_config["catalog_url"])
        model = open_copernicus_catalog(cat)
        console.log("successfully opened reference model")

        status.update(
            "[bold blue]compare temperature from reference model and tag log[/]"
        )
        differences = subtract_data(tag, model, parameters)
        differences.chunk({"time": 1, "lat": -1, "lon": -1}).to_zarr(
            f"{target_root}/diff.zarr", mode="w", consolidated=True
        )
        console.log("stored temperature differences")

        # open back the diff
        differences = (
            xr.open_dataset(f"{target_root}/diff.zarr", engine="zarr", chunks={})
            .pipe(lambda ds: ds.merge(ds[["latitude", "longitude"]].compute()))
            .swap_dims({"lat": "yi", "lon": "xi"})
            .drop_vars(["lat", "lon"])
        )
        console.log("reopened temperature differences")

        status.update("[bold blue]verifying result[/]")
        counts = differences["diff"].count(["xi", "yi"]).compute()
        console.log("finished detecting missing time slices")
        if (counts == 0).any():
            raise click.ClickException(
                "some time slices are 0. Try rerunning the step or"
                " checking the connection to the data server."
            )
        console.log("detecting missing time slices: none found")

        status.update("[bold blue]regridding[/]")
        regridded = regrid(differences, parameters)
        regridded.chunk({"x": -1, "y": -1, "time": 1}).to_zarr(
            f"{target_root}/diff-regridded.zarr",
            mode="w",
            consolidated=True,
        )
        console.log("finished regridding")

        # temperature emission matrices
        differences = xr.open_dataset(
            f"{target_root}/diff-regridded.zarr", engine="zarr", chunks={}
        )

        status.update("[bold blue]constructing emission matrices from temperature[/]")
        emission = temperature_emission_matrices(differences, tag, parameters)
        emission.chunk({"x": -1, "y": -1, "time": 1}).to_zarr(
            f"{target_root}/emission.zarr",
            mode="w",
            consolidated=True,
        )
        console.log("finished constructing emission matrices from temperature data")

        del differences

        # acoustic emission matrices
        emission = xr.open_dataset(
            f"{target_root}/emission.zarr", engine="zarr", chunks={}
        )
        status.update(
            "[bold blue]constructing emission matrices from acoustic detections[/]"
        )
        combined = emission.merge(
            acoustic.emission_probability(
                tag,
                emission[["time", "cell_ids", "mask"]].compute(),
                parameters["receiver_buffer"],
            )
        )

        combined.chunk({"x": -1, "y": -1, "time": 1}).to_zarr(
            f"{target_root}/emission-acoustic.zarr", mode="w", consolidated=True
        )
        console.log("finished writing emission matrices from acoustic detections")

        del combined


@main.command("estimate", short_help="estimate the model parameter")
@click.option("--cluster-definition", type=click.File(mode="r"))
@click.option(
    "--compute/--no-compute",
    type=bool,
    default=True,
    help="load the emission pdf into memory before the parameter estimation",
)
@click.option(
    "--estimator",
    type=click.Choice(["cached", "eager"]),
    default="cached",
    help="choose the estimator",
)
@click.argument("parameters", type=click.File(mode="r"))
@click.argument("runtime_config", type=click.File(mode="r"))
def estimate(parameters, runtime_config, cluster_definition, estimator, compute):
    from pangeo_fish.hmm.optimize import EagerBoundsSearch

    runtime_config = json.load(runtime_config)
    parameters = json.load(parameters, object_hook=decode_parameters)
    cluster_definition = json.load(cluster_definition)

    target_root = construct_target_root(runtime_config, parameters)
    cache_path = target_root / "cache.zarr"

    if compute:
        chunks = None
        client = nullcontext()
    else:
        chunks = {"x": -1, "y": -1}
        client = create_cluster(**cluster_definition)
        console.print(f"dashboard link: {client.dashboard_link}")

    estimators = {
        "cached": partial(create_cached_estimator, cache_path),
        "eager": EagerEstimator,
    }

    with (
        client,
        console.status("[bold blue]estimating the model parameter...[/]") as status,
    ):
        emission = (
            xr.open_dataset(
                f"{target_root}/emission-acoustic.zarr",
                engine="zarr",
                chunks=chunks,
                inline_array=True,
            )
            .pipe(combine_emission_pdf)
            .pipe(maybe_compute, compute=compute)
        )
        console.log("opened emission probabilities")

        console.status("[bold blue]detecting missing timesteps[/]")
        counts = emission["pdf"].count(["y", "x"]).compute()
        if (counts == 0).any():
            raise click.ClickException(
                "Some time slices are all-nan, which will cause the optimization to fail."
                " This can happen if a component of the emission probability matrices has"
                " all-nan time slices, or if the components don't have overlaps in"
                " all-nan areas."
            )
        console.log("detecting missing timesteps: none found")

        estimator_ = estimators.get(estimator)()
        optimizer = EagerBoundsSearch(
            estimator_,
            (1e-4, emission.attrs["max_sigma"]),
            optimizer_kwargs={"disp": 3, "xtol": parameters.get("tolerance", 0.01)},
        )

        status.update("[bold blue]searching for optimal model parameters[/]")
        optimized = optimizer.fit(emission)
        console.log("model parameter: completed search")

        if optimized.sigma == emission.attrs["max_sigma"]:
            raise click.ClickException(
                "Found the upper limit of the parameter search space."
                " Make sure the search space is big enough."
            )
        console.log("model parameter: checks passed")

        status.update("[bold blue]storing the optimized model parameter[/]")
        params = optimized.to_dict()
        with target_root.joinpath("parameters.json").open(mode="w") as f:
            json.dump(params, f)
        console.log("model parameter: finished writing the optimized model parameter")


@main.command("decode", short_help="produce the model output")
@click.option("--cluster-definition", type=click.File(mode="r"))
@click.option(
    "--estimator",
    type=click.Choice(["cached", "eager"]),
    default="cached",
    help="choose the estimator",
)
@click.argument("parameters", type=click.File(mode="r"))
@click.argument("runtime_config", type=click.File(mode="r"))
def decode(parameters, runtime_config, cluster_definition, estimator):
    # read input data: emission, sigma
    from pangeo_fish.io import save_trajectories

    runtime_config = json.load(runtime_config)
    parameters = json.load(parameters, object_hook=decode_parameters)
    cluster_definition = json.load(cluster_definition)

    target_root = construct_target_root(runtime_config, parameters)
    cache_path = target_root / "cache.zarr"
    tracks_root = target_root / "tracks"
    tracks_root.mkdir(exist_ok=True, parents=True)

    estimators = {
        "cached": partial(create_cached_estimator, cache_path),
        "eager": EagerEstimator,
    }

    with (
        create_cluster(**cluster_definition) as client,
        console.status("[bold blue]decoding...[/]") as status,
    ):
        console.print(f"dashboard link: {client.dashboard_link}")

        emission = (
            xr.open_dataset(
                f"{target_root}/emission-acoustic.zarr",
                engine="zarr",
                chunks={"x": -1, "y": -1},
                inline_array=True,
            ).pipe(combine_emission_pdf)
            # .pipe(maybe_compute, compute=compute)
        )
        console.log("opened emission probabilities")

        with target_root.joinpath("parameters.json").open(mode="r") as f:
            params = json.load(f)
        console.log("read model parameter")

        optimized = estimators.get(estimator)(**params)
        console.log("created the estimator")

        status.update("[bold blue]predicting the state probabilities...[/]")
        states = optimized.predict_proba(emission)
        console.log("constructed task graph")
        states.chunk({"time": 1, "x": -1, "y": -1}).to_zarr(
            f"{target_root}/states.zarr", mode="w", consolidated=True
        )
        console.log("finished writing the state probabilities")

        states = xr.open_dataset(
            f"{target_root}/states.zarr", engine="zarr", chunks={}, inline_array=True
        )
        console.log("reopened the state probabilities")

        status.update("[bold blue]decoding tracks[/]")
        trajectories = optimized.decode(
            emission,
            states,
            mode=parameters["track_modes"],
            progress=True,
            additional_quantities=parameters["additional_track_quantities"],
        )
        console.log("tracks: computed successfully")

        save_trajectories(trajectories, tracks_root, format="parquet")
        console.log("tracks: stored to disk")


@main.command("visualize", short_help="visualize the decoded results")
@click.option("--cluster-definition", type=click.File(mode="r"))
@click.option(
    "--compute/--no-compute",
    type=bool,
    default=True,
    help="load the emission pdf into memory before the parameter estimation",
)
@click.argument("parameters", type=click.File(mode="r"))
@click.argument("runtime_config", type=click.File(mode="r"))
def visualize(parameters, runtime_config, cluster_definition, compute):
    import cmocean  # noqa: F401
    import holoviews as hv
    import hvplot.xarray  # noqa: F401
    import xmovie

    from pangeo_fish import visualization
    from pangeo_fish.io import read_trajectories

    hv.extension("matplotlib")

    runtime_config = json.load(runtime_config)
    parameters = json.load(parameters, object_hook=decode_parameters)
    cluster_definition = json.load(cluster_definition)

    viz_params = parameters.get("visualization", {})
    movie_params = viz_params.get("movie", {})
    track_params = viz_params.get("tracks", {})

    target_root = construct_target_root(runtime_config, parameters)
    tracks_root = target_root / "tracks"
    viz_root = target_root / "plots"
    viz_root.mkdir(exist_ok=True, parents=True)

    if compute:
        chunks = None
        client = nullcontext()
    else:
        chunks = {"x": -1, "y": -1}
        client = create_cluster(**cluster_definition)
        console.print(f"dashboard link: {client.dashboard_link}")

    with client, console.status("[bold blue]visualizing the results...[/]") as status:
        console.print(f"dashboard link: {client.dashboard_link}")

        status.update("[bold blue]tracks:[/] reading tracks")
        trajectories = read_trajectories(
            tracks_root, parameters["track_modes"], format="parquet"
        )
        console.log("tracks: reading successful")

        cmap = track_params.get("cmap", "cmo.speed")
        tiles = track_params.get("tiles", "CartoLight")
        format = track_params.get("format", "png")

        status.update("[bold blue]tracks:[/] plotting")
        plot = visualization.plot_trajectories(
            trajectories, subplots=False, c="speed", tiles=tiles, cmap=cmap
        )
        hv.save(plot, f"{viz_root}/tracks_combined.{format}", fmt=format)
        console.log("tracks: done plotting")

        status.update("[bold blue]tracks:[/] plotting with multiple subplots")
        plot = visualization.plot_trajectories(
            trajectories, subplots=True, c="speed", tiles=tiles, cmap=cmap
        )
        hv.save(plot, f"{viz_root}/tracks.{format}", fmt=format)
        console.log("tracks: done plotting with subplots")

        status.update(
            "[bold blue]movie:[/] opening and combining state and emission probabilities"
        )
        emission = (
            xr.open_dataset(
                f"{target_root}/emission-acoustic.zarr",
                engine="zarr",
                chunks=chunks,
                inline_array=True,
            )
            .pipe(combine_emission_pdf)
            .rename_vars({"pdf": "emission"})
            .drop_vars(["final", "initial"])
        )
        states = xr.open_dataset(
            f"{target_root}/states.zarr",
            engine="zarr",
            chunks=chunks,
            inline_array=True,
        ).where(emission["mask"].notnull())
        data = xr.merge([states, emission.drop_vars(["mask"])]).pipe(
            maybe_compute, compute=compute
        )
        console.log("successfully combined state and emission probabilities")

        status.update("[bold blue]movie:[/] creating definition")
        height = movie_params.get("height", 15)
        width = movie_params.get("width", 12)
        dpi = movie_params.get("dpi", 300)
        format = movie_params.get("format", "mp4")
        mov = xmovie.Movie(
            (
                data.pipe(
                    lambda ds: ds.merge(ds[["longitude", "latitude"]].compute())
                ).pipe(visualization.filter_by_states)
            ),
            plotfunc=visualization.create_frame,
            input_check=False,
            pixelwidth=width * dpi,
            pixelheight=height * dpi,
            dpi=dpi,
        )
        console.log("movie: definition")
        status.update("[bold blue]movie:[/] creating frames and saving")
        mov.save(
            f"{target_root}/states.{format}",
            parallel=True,
            overwrite_existing=True,
        )
        console.log("movie: stored to disk")
