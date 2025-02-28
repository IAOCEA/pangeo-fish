"""Module for managing I/O operations."""

import json
import warnings

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


def read_stations(f):
    return pd.read_csv(
        f,
        parse_dates=["deploy_time", "recover_time"],
        index_col="deployment_id",
    ).pipe(tz_convert, {"deploy_time": None, "recover_time": None})


def open_tag(root, name, storage_options=None):
    """open a tag

    Parameters
    ----------
    root : str or fsspec.FSMap
        The tag root. If not a mapper object, `storage_options` need
        to contain the necessary access options.
    name : str
        The DST name of the tag.
    storage_options : mapping, optional
        The storage options required to open the mapper. Only used if `root` is a url.

    Returns
    -------
    tag : xarray.DataTree
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
        stations = read_stations(mapper.dirfs.open("stations.csv"))
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

    return xr.DataTree.from_dict(mapping)


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
        default, the chunksizes are `{"lat": -1, "lon": -1, "depth": 11, "time": 8}`

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


def prepare_dataset(dataset, chunks=None, bbox=None, names=None):
    """Prepares a dataset of a reference model.
    It renames some variables (see ``names``), adds dynamic bathymetry and depth and broadcast lat(itude)/lon(gitude) coordinates.

    Parameters
    ----------
    dataset : xarray.Dataset
        The pre-opened dataset of the fields data
    chunks : mapping, optional
        The initial chunk size. Should be multiples of the on-disk chunk sizes. By
        default, the chunksizes are ``{"lat": -1, "lon": -1, "depth": 11, "time": 8}``
    bbox : dict[str, tuple[float, float]], optional
        The spatial boundaries of the area of interest. Shoud have the keys "longitude" and "latitude".
        If provided, it checks whether there is data available within the dataset for the area.
    names : dict[str, str], optional
        A dictionary that maps the three variables that correspond to the "TEMP", "XE" and "H0" data. By
        default, the names align data from the Copernicus Marine Service with ``{"thetao": "TEMP", "zos": "XE", "deptho": "H0"}``.

    Returns
    -------
    ds : xarray.Dataset
        The post-processed dataset.
    """
    # checks that the studied area is included in the dataset
    if bbox is not None:
        if not all([k in ["latitude", "longitude"] for k in bbox.keys()]):
            raise ValueError(
                'The "bbox" argument must have the keys "latitude" and "longitude".'
            )

        box = {
            "latitude": [
                dataset.lat.min().to_numpy().item(),
                dataset.lat.max().to_numpy().item(),
            ],
            "longitude": [
                dataset.lon.min().to_numpy().item(),
                dataset.lon.max().to_numpy().item(),
            ],
        }
        valid_bbox = True

        for inter in ["latitude", "longitude"]:
            tmp1 = box[inter]  # field model
            tmp2 = bbox[inter]
            if (tmp2[0] < tmp1[0]) or (tmp2[1] > tmp1[1]):
                valid_bbox = False

        if not valid_bbox:
            warnings.warn(
                "The studied area is not entirely included in the dataset!", UserWarning
            )

    if chunks is None:
        chunks = {"lat": -1, "lon": -1, "depth": 11, "time": 8}

    if names is None:
        names = {"thetao": "TEMP", "zos": "XE", "deptho": "H0"}

    coords_and_vars = [v for v in dataset.variables]
    if not all([n in coords_and_vars for n in names.keys()]):
        raise ValueError(
            f"The dataset does not include all the variables indexed in ``names``: {list(names.keys())}."
        )

    if not all([k in names.values() for k in ["TEMP", "XE", "H0"]]):
        raise ValueError(
            'The mapping ``names`` must have values "TEMP", "XE" and "H0".'
        )

    ds = (
        dataset.chunk(chunks=chunks)
        .rename(names)
        # .assign_coords({"time": lambda ds: ds["time"].astype("datetime64[ns]")}) # useless?
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
        .pipe(broadcast_variables, {"lat": "latitude", "lon": "longitude"})  # useless?
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
    """save a Holoviews plot to an HTML file either locally or on an S3 bucket.

    Parameters:
    -----------
    plot :
        Holoviews plot object.
    filepath : str
        The file path where the plot HTML file will be saved. If the file path starts with 's3://', the plot will be saved to an S3 bucket.
    storage_options : (dict, optional)
        Dictionary containing storage options for connecting to the S3 bucket (required if saving to S3).

    Returns:
    --------
    success : bool
        True if the plot was saved successfully, False otherwise.
    message : str
        A message describing the outcome of the operation.
    """

    import hvplot
    import hvplot.xarray

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
    model="GLOBAL_ANALYSISFORECAST_PHY_001_024",
    format="geoChunked",
    freq="D",
    interp_thetao=False,
):
    """
    Retrieve Copernicus Marine data in zarr format.

    .. warning::
        This function is not fully finalized and may require further adjustments.


    Parameters:
    -----------
    name : str
        Name of the dataset to retrieve. Supported models and corresponding frequencies are:
            - "GLOBAL_ANALYSISFORECAST_PHY_001_024" with freq = "D" (daily)
                working.
            - "NWSHELF_ANALYSISFORECAST_PHY_004_013" with freq = "D" (daily)
                working.

            Future supported models and corresponding frequencies are:
                - "GLOBAL_ANALYSISFORECAST_PHY_001_024" with freq =  "H" (hourly)
                    not working. (ValueError: conflicting sizes for dimension 'time': length 20184 on 'XE' and length 3365 on {'elevation': 'elevation', 'latitude': 'latitude', 'longitude': 'longitude', 'time': 'time'})
                - "GLOBAL_MULTIYEAR_PHY_001_030" with freq = "NEW" or "OLD",
                    not working. (KeyError: 'cmems_mod_glo_phy_anfc_0.083deg_static_202211--ext--bathy')
                - "IBI_MULTIYEAR_PHY_005_002" with freq = "D" (daily),
                    not working(KeyError: 'cmems_mod_ibi_phy_anfc_0.027deg-3D_P1D-m_202211')
                - "IBI_MULTIYEAR_PHY_005_002" with freq =  "H" (hourly),
                    not working (KeyError: 'cmems_mod_ibi_phy_anfc_0.027deg-3D_PT1H-m_202211')
                - "IBI_ANALYSISFORECAST_PHY_005_001" with freq = "D" (daily)
                    (KeyError: 'cmems_mod_ibi_phy_my_0.083deg-3D_P1D-m_202012')
                - "IBI_ANALYSISFORECAST_PHY_005_001" with freq = "H" (hourly),
                    (KeyError: '')
                - "NWSHELF_ANALYSISFORECAST_PHY_004_013" with freq = "H" (hourly),
                    (ValueError: conflicting sizes for dimension 'time': length 26616 on 'XE' and length 10560 on {'elevation': 'elevation', 'latitude': 'latitude', 'longitude': 'longitude', 'time': 'time'})
                - "NWSHELF_MULTIYEAR_PHY_004_009" with freq = "H" (hourly) .
                    (KeyError: '')
                - "NWSHELF_MULTIYEAR_PHY_004_009" with freq = "D" (daily).
                    (KeyError: 'static')

    format : {"arco-geo-series", "arco-time-series"}, default: "arco-geo-series"
        Format of the dataset.

    Returns
    -------
    xarray.Dataset
        Dataset containing retrieved data.
    """
    # Add here datas which are valid.

    name = {
        "GLOBAL_ANALYSISFORECAST_PHY_001_024": {
            "thetao": {
                "H": "cmems_mod_glo_phy-thetao_anfc_0.083deg_PT6H-i_202406",
                "D": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m_202406",
            },
            "zos": {
                "H": "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m_202406",
                "D": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m_202406",
            },
            "deptho": "cmems_mod_glo_phy_anfc_0.083deg_static_202211--ext--bathy",
        },
        "GLOBAL_MULTIYEAR_PHY_001_030": {
            "thetao": {
                "NEW": "cmems_mod_glo_phy_myint_0.083deg_P1D-m_202311",
                "OLD": "cmems_mod_glo_phy_my_0.083deg_P1D-m_202311",
            },
            "zos": {
                "NEW": "cmems_mod_glo_phy_myint_0.083deg_P1D-m_202311",
                "OLD": "cmems_mod_glo_phy_anfc_0.083deg_P1D-m_202406",
            },
            "deptho": "cmems_mod_glo_phy_anfc_0.083deg_static_202211--ext--bathy",
        },
        "IBI_MULTIYEAR_PHY_005_002": {
            "thetao": {
                "H": "cmems_mod_ibi_phy_anfc_0.027deg-3D_PT1H-m_202211",
                "D": "cmems_mod_ibi_phy_anfc_0.027deg-3D_P1D-m_202211",
            },
            "zos": {
                "H": "cmems_mod_ibi_phy_anfc_0.027deg-2D_PT1H-m_202211",
                "D": "cmems_mod_ibi_phy_anfc_0.027deg-3D_P1D-m_202211",
            },
            "deptho": "cmems_mod_ibi_phy_anfc_0.027deg-3D_static_202211--ext--bathy",
        },
        "IBI_ANALYSISFORECAST_PHY_005_001": {
            "thetao": {"H": "", "D": "cmems_mod_ibi_phy_my_0.083deg-3D_P1D-m_202012"},
            "zos": {
                "H": "cmems_mod_ibi_phy_my_0.083deg-2D_PT1H-m_202012",
                "D": "cmems_mod_ibi_phy_my_0.083deg-3D_P1D-m_202012",
            },
            "deptho": "cmems_mod_ibi_phy_my_0.083deg-3D_static_202012--ext--bathy",
        },
        "NWSHELF_ANALYSISFORECAST_PHY_004_013": {
            "thetao": {
                "H": "cmems_mod_nws_phy_anfc_0.027deg-3D_PT1H-m_202309",
                "D": "cmems_mod_nws_phy_anfc_0.027deg-3D_P1D-m_202309",
            },
            "zos": {
                "H": "cmems_mod_nws_phy_anfc_0.027deg-2D_PT1H-m_202309",
                "D": "cmems_mod_nws_phy_anfc_0.027deg-3D_P1D-m_202309",
            },
            "deptho": "cmems_mod_nws_phy_anfc_0.027deg-3D_static_202309--ext--bathy",
        },
        "NWSHELF_MULTIYEAR_PHY_004_009": {
            "thetao": {"H": "", "D": "cmems_mod_nws_phy-t_my_7km-3D_P1D-m_202012"},
            "zos": {"H": "", "D": "cmems_mod_nws_phy-ssh_my_7km-2D_P1D-m_202012"},
            "deptho": "cmems_mod_nws_phy-bottomt_my_7km-2D_P1D-m_202012",
        },
    }
    ##TODO, Here in the stac catalogue, we will need to add the data copied in GFTS
    #    import copernicusmarine as copernicusmarine

    import pystac_client

    client = pystac_client.Client.open("https://keewis-copernicus-marine.hf.space")
    result = client.search(
        collections=[model],
    ).item_collection()
    var = {item.id: item for item in result.items}

    # Open necessary datasets

    asset = var[name[model]["thetao"][freq]].assets["geoChunked"]
    thetao = xr.open_dataset(asset.href, engine="zarr", chunks={}).thetao.to_dataset()

    asset = var[name[model]["zos"][freq]].assets["geoChunked"]
    zos = xr.open_dataset(asset.href, engine="zarr", chunks={}).zos

    asset = var[name[model]["deptho"]].assets["static"]
    deptho = xr.open_dataset(asset.href, engine="zarr", chunks={}).deptho

    if interp_thetao:
        thetao = thetao.interp(time=zos.time, method="quadratic")

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
