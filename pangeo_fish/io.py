import io
import json
import os

import datatree
import fsspec
import geopandas as gpd
import movingpandas as mpd
import pandas as pd
import xarray as xr

from pangeo_fish.dataset_utils import broadcast_variables


def tz_convert(df, timezones):
    """Convert the timezone of columns in a dataframe

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe.
    timezones : mapping of str to str
        The time zones to convert to per column.
    """
    new_columns = {
        column: pd.Index(df[column]).tz_convert(tz) for column, tz in timezones.items()
    }

    return df.assign(**new_columns)


def read_tag_database(url):
    """read the tag database

    Parameters
    ----------
    url : str, path-like or file-like
        The file path, url, path-like or pre-opened file object of the file to open

    Returns
    -------
    tag_database : pandas.DataFrame
        The opened tag database
    """
    return pd.read_csv(url, sep=";")


def read_detection_database(url):
    """read the detection database

    Parameters
    ----------
    url : str, path-like or file-like
        The file path, url, path-like or pre-opened file object of the file to open

    Returns
    -------
    detection_database : pandas.DataFrame
        The opened detection database

    """
    if isinstance(url, os.PathLike):
        url = os.fsspath(url)

    if isinstance(url, str):
        opened = fsspec.open(url, mode="r")
    else:
        opened = url

    with opened as f:
        # work around the weird quoting
        lines = (line.replace('"', "") for line in f)
        data = "\n".join(lines)

    content = io.StringIO(data)
    return (
        pd.read_csv(content, parse_dates=[1])
        .rename(columns={"date_time": "time"})
        .set_index("time")
    )


def open_tag(root, name, storage_options=None):
    """open a tag

    Parameters
    ----------
    root : str or fsspec.FSMap
        The tag root. If not a mapper object, ``storage_options`` need
        to contain the necessary access options.
    name : str
        The DST name of the tag.
    storage_options : mapping, optional
        The storage options required to open the mapper. Only used if ``root`` is a url.

    Returns
    -------
    tag : datatree.DataTree
        The opened tag with involved stations, acoustic tags, tag log and metadata
    """
    if isinstance(root, str):
        if storage_options is None:
            storage_options = {}
        mapper = fsspec.get_mapper(root, **storage_options)
    else:
        mapper = root

    dst = pd.read_csv(
        mapper.dirfs.open(f"{name}/dst.csv"), parse_dates=["time"], index_col="time"
    ).tz_convert(None)

    tagging_events = pd.read_csv(
        mapper.dirfs.open(f"{name}/tagging_events.csv"),
        parse_dates=["time"],
        index_col="event_name",
    ).pipe(tz_convert, {"time": None})

    metadata = json.load(mapper.dirfs.open(f"{name}/metadata.json"))

    mapping = {
        "/": xr.Dataset(attrs=metadata),
        "dst": dst.to_xarray(),
        "tagging_events": tagging_events.to_xarray(),
    }
    if mapper.dirfs.exists("stations.csv"):
        stations = pd.read_csv(
            mapper.dirfs.open("stations.csv"),
            parse_dates=["deploy_time", "recover_time"],
            index_col="deployment_id",
        ).pipe(tz_convert, {"deploy_time": None, "recover_time": None})
        if len(stations) > 0:
            mapping["stations"] = stations.to_xarray()

    if mapper.dirfs.exists(f"{name}/acoustic.csv"):
        acoustic = pd.read_csv(
            mapper.dirfs.open(f"{name}/acoustic.csv"),
            parse_dates=["time"],
            index_col="time",
        ).tz_convert(None)
        if len(acoustic) > 0:
            mapping["acoustic"] = acoustic.to_xarray()

    return datatree.DataTree.from_dict(mapping)


def open_copernicus_catalog(cat, chunks=None):
    """assemble the given intake catalog into a dataset

    .. warning::
        This will only work for the catalog at https://data-taos.ifremer.fr/references/copernicus.yaml

    Parameters
    ----------
    cat : intake.Catalog
        The pre-opened intake catalog
    chunks : mapping, optional
        The initial chunk size. Should be multiples of the on-disk chunk sizes. By
        default, the chunksizes are ``{"lat": -1, "lon": -1, "depth": 11, "time": 8}``

    Returns
    -------
    ds : xarray.Dataset
        The assembled dataset.
    """
    if chunks is None:
        chunks = {"lat": -1, "lon": -1, "depth": 11, "time": 8}

    ds = (
        cat.data(type="TEM", chunks=chunks)
        .to_dask()
        .rename({"thetao": "TEMP"})
        .get(["TEMP"])
        .assign_coords({"time": lambda ds: ds["time"].astype("datetime64[ns]")})
        .assign(
            {
                "XE": cat.data(type="SSH", chunks=chunks).to_dask().get("zos"),
                "H0": (
                    cat.data_tmp(type="mdt", chunks=chunks)
                    .to_dask()
                    .get("deptho")
                    .rename({"latitude": "lat", "longitude": "lon"})
                ),
                "mask": (
                    cat.data_tmp(type="mdt", chunks=chunks)
                    .to_dask()
                    .get("mask")
                    .rename({"latitude": "lat", "longitude": "lon"})
                ),
            }
        )
        # TODO: figure out the definition of `depth` and if there are standard names for these
        .assign(
            {
                "dynamic_depth": lambda ds: (ds["depth"] + ds["XE"]).assign_attrs(
                    {"units": "m", "positive": "down"}
                ),
                "dynamic_bathymetry": lambda ds: (ds["H0"] + ds["XE"]).assign_attrs(
                    {"units": "m", "positive": "down"}
                ),
            }
        )
        .pipe(broadcast_variables, {"lat": "latitude", "lon": "longitude"})
    )

    return ds


def save_trajectories(traj, root, storage_options=None, format="geoparquet"):
    from pangeo_fish.tracks import to_dataframe

    converters = {
        "geoparquet": lambda x: x.drop(columns="traj_id"),
        "parquet": to_dataframe,
    }
    converter = converters.get(format)
    if converter is None:
        raise ValueError(f"unknown format: {format!r}")

    if not isinstance(root, str):
        root = str(root)

    trajectories = getattr(traj, "trajectories", [traj])

    for traj in trajectories:
        path = f"{root}/{traj.id}.parquet"

        df = converter(traj.df)
        df.to_parquet(path, storage_options=storage_options)


def read_trajectories(names, root, storage_options=None, format="geoparquet"):
    """read trajectories from disk

    Parameters
    ----------
    root : str or path-like
        The root directory containing the track files.
    names : list of str
        The names of the tracks to read.
    format : {"parquet", "geoparquet"}, default: "geoparquet"
        The format of the files.

    Returns
    -------
    mpd.TrajectoryCollection
        The read tracks as a collection.
    """

    def read_geoparquet(root, name, storage_options):
        path = f"{root}/{name}.parquet"

        gdf = gpd.read_parquet(path, storage_options=storage_options)

        return mpd.Trajectory(gdf, name)

    def read_parquet(root, name):
        path = f"{root}/{name}.parquet"

        df = pd.read_parquet(path, storage_options=storage_options)

        return mpd.Trajectory(df, name, x="longitude", y="latitude")

    readers = {
        "geoparquet": read_geoparquet,
        "parquet": read_parquet,
    }

    reader = readers.get(format)
    if reader is None:
        raise ValueError(f"unknown format: {format}")

    return mpd.TrajectoryCollection([reader(root, name) for name in names])


def save_html_hvplot(plot, filepath, storage_options=None):
    import hvplot
    import hvplot.xarray

    """
    Save a Holoviews plot to an HTML file either locally or on an S3 bucket.

    Parameters:
    - plot: Holoviews plot object.
    - filepath (str): The file path where the plot HTML file will be saved. If the file path starts with 's3://', the plot will be saved to an S3 bucket.
    - storage_options (dict, optional): Dictionary containing storage options for connecting to the S3 bucket (required if saving to S3).

    Returns:
    - success (bool): True if the plot was saved successfully, False otherwise.
    - message (str): A message describing the outcome of the operation.
    """
    try:
        if filepath.startswith("s3://"):
            import s3fs

            if storage_options is None:
                raise ValueError("Storage options must be provided for S3 storage.")

            s3 = s3fs.S3FileSystem(**storage_options)
            with s3.open(filepath, "w") as f:
                hvplot.save(plot, f)
        else:
            hvplot.save(plot, filepath)

        return True, "Plot saved successfully."

    except Exception as e:
        return False, f"Error occurred: {str(e)}"


def open_copernicus_zarr(
    name="cmems_mod_ibi_phy_my_0.083deg", format="arco-geo-series"
):
    """
    Retrieve Copernicus Marine data in zarr format.

    .. warning::
        This function is not fully finalized and may require further adjustments.


    Parameters:
    -----------
    name : str, default: "cmems_mod_ibi_phy_my_0.083deg"
        Name of the dataset to retrieve.
    format : {"arco-geo-series", "arco-time-series"}, default: "arco-geo-series"
        Format of the dataset.

    Returns
    -------
    xarray.Dataset
        Dataset containing retrieved data.
    """
    import copernicusmarine as copernicusmarine

    # Dictionary to store URIs by key
    uris_by_key = {}

    # Retrieve dataset information from Copernicus Marine
    catalogue = copernicusmarine.describe(
        include_datasets=True,
        contains=[name],
    )

    # Iterate through datasets in the catalogue
    for value in catalogue["products"][0]["datasets"]:
        dataset_id = value["dataset_id"]

        # Check if the dataset contains relevant data
        if any(
            substring in dataset_id
            for substring in ["static", "2D_PT1H-m", "3D_PT1H-m", "3D_P1D-m"]
        ):
            uris = []

            # Extract URIs for relevant services
            for service in value["versions"][0]["parts"][0]["services"]:
                service_name = service["service_type"]["service_name"]
                if service_name in [format, "arco-time-series", "static-arco"]:
                    uris.append(service["uri"])

            uris_by_key[dataset_id] = uris

    # Open necessary datasets
    thetao = xr.open_dataset(
        uris_by_key[f"{name}-3D_P1D-m"][0], engine="zarr", chunks={}
    )[["thetao"]]
    zos = xr.open_dataset(uris_by_key[f"{name}-3D_P1D-m"][0], engine="zarr", chunks={})[
        "zos"
    ]
    deptho = xr.open_dataset(
        uris_by_key[f"{name}-3D_static"][0], engine="zarr", chunks={}
    )["deptho"]

    ds = (
        # assemble dataset
        thetao.rename({"thetao": "TEMP"})
        .assign(
            {
                "XE": zos.variable,
                "H0": deptho.variable,
                "mask": deptho.isnull().variable,
            }
        )
        .rename({"latitude": "lat", "longitude": "lon", "elevation": "depth"})
        # Rearrange depth coordinates
        .assign(depth=lambda ds: abs(ds["depth"]))
        .isel(depth=slice(None, None, -1))
        # assign dynamic depth and bathymetry
        .assign(
            {
                "dynamic_depth": lambda ds: (ds["depth"] + ds["XE"]).assign_attrs(
                    {"units": "m", "positive": "down"}
                ),
                "dynamic_bathymetry": lambda ds: (ds["H0"] + ds["XE"]).assign_attrs(
                    {"units": "m", "positive": "down"}
                ),
            }
        )
        .pipe(broadcast_variables, {"lat": "latitude", "lon": "longitude"})
    )

    return ds
