import itertools
import json
import os
from dataclasses import dataclass, field
from functools import partial

import dask
import healpy as hp
import numba
import numpy as np
import sparse
import xarray as xr


def geographic_to_astronomic(lat, lon, rot):
    """Transform geographic coordinates to astronomic coordinates

    Parameters
    ----------
    lat : array-like
        geographic latitude, in degrees
    lon : array-like
        geographic longitude, in degrees
    rot : array-like
        Two element list with the rotation transformation (shift?) used by the grid, in
        degrees

    Returns
    -------
    theta : array-like
        Colatitude in degrees
    phi : array-like
        Astronomic longitude in degrees
    """
    theta = 90.0 - lat - rot["lat"]
    phi = -lon + rot["lon"]

    return theta, phi


def astronomic_to_cartesian(theta, phi, dim="receiver_id"):
    """Transform astronomic coordinates to cartesian coordinates

    Parameters
    ----------
    theta : array-like
        astronomic colatitude, in degrees
    phi : array-like
        astronomic longitude, in degrees
    dim : hashable
        Name of the dimension

    Returns
    -------
    cartesian : xarray.Dataset
        Cartesian coordinates

    See Also
    --------
    healpy.ang2vec
    """
    # TODO: try to determine `dim` automatically
    cartesian = xr.apply_ufunc(
        hp.ang2vec,
        np.deg2rad(theta),
        np.deg2rad(phi),
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim, "cartesian"]],
    )

    return cartesian.assign_coords(cartesian=["x", "y", "z"])


def astronomic_to_cell_ids(nside, phi, theta):
    """Compute cell ids from astronomic coordinates

    Parameters
    ----------
    nside : int
        Healpix resolution level
    phi, theta : array-like
        astronomic longitude and colatitude, in degrees

    Returns
    -------
    cell_ids : xarray.DataArray
        The computed cell ids
    """
    phi_, theta_ = dask.compute(phi, theta)

    cell_ids = xr.apply_ufunc(
        hp.ang2pix,
        nside,
        np.deg2rad(theta_),
        np.deg2rad(phi_),
        kwargs={"nest": True},
        input_core_dims=[[], ["x", "y"], ["x", "y"]],
        output_core_dims=[["x", "y"]],
    )

    return cell_ids


@numba.jit(nopython=True, parallel=True)
def _compute_indices(nside):
    nstep = int(np.log2(nside))

    lidx = np.arange(nside**2)
    xx = np.zeros_like(lidx)
    yy = np.zeros_like(lidx)

    for i in range(nstep):
        p1 = 2**i
        p2 = (lidx // 4**i) % 4

        xx = xx + p1 * (p2 % 2)
        yy = yy + p1 * (p2 // 2)

    return xx, yy


def _compute_coords(nside):
    lidx = np.arange(nside**2)
    theta, phi = hp.pix2ang(nside, lidx, nest=True)

    lat = 90.0 - np.rad2deg(theta)
    lon = -np.rad2deg(phi)

    return lat, lon, lidx


@dataclass
class HealpyGridInfo:
    """Class representing a HealPix grid

    Attributes
    ----------
    nside : int
        HealPix grid resolution
    rot : dict of str to float
        Rotation of the healpix sphere.
    coords : xarray.Dataset
        Unstructured grid coordinates: latitude, longitude, cell ids.
    indices ; xarray.DataArray
        Indices that can be used to reorder to a flattened 2D healpy grid
    """

    nside: int

    rot: dict[str, float]

    indices: xr.DataArray = field(repr=False)
    coords: xr.Dataset = field(repr=False)

    def unstructured_to_2d(
        self, unstructured, dim="cells", keep_attrs="drop_conflicts"
    ):
        def _unstructured_to_2d(unstructured, indices, new_shape):
            if unstructured.size in (0, 1):
                return np.ones((1, 1), dtype=unstructured.dtype)

            return unstructured[..., indices].reshape(new_shape)

        new_sizes = {"x": self.nside, "y": self.nside}

        return xr.apply_ufunc(
            _unstructured_to_2d,
            unstructured,
            self.indices,
            input_core_dims=[[dim], ["cells"]],
            output_core_dims=[["x", "y"]],
            kwargs={"new_shape": tuple(new_sizes.values())},
            dask="parallelized",
            dask_gufunc_kwargs={"output_sizes": new_sizes},
            vectorize=True,
            keep_attrs=keep_attrs,
        )

    def to_xarray(self):
        attrs = {"nside": self.nside} | {f"rot_{k}": v for k, v in self.rot.items()}

        return self.coords.assign_attrs(attrs)


def concat(iterable):
    return itertools.chain.from_iterable(iterable)


def unique(iterable):
    return list(dict.fromkeys(iterable))


def create_grid(nside, rot={"lat": 0, "lon": 0}):
    xx, yy = _compute_indices(nside)

    raw_indices = np.full((nside, nside), fill_value=-1, dtype=int)
    raw_indices[xx, yy] = np.arange(nside**2)
    indices = xr.DataArray(np.ravel(raw_indices), dims="cells").chunk()

    lat_, lon_, cell_ids = _compute_coords(nside)
    lat = lat_ - rot["lat"]
    lon = lon_ + rot["lon"]

    resolution = hp.nside2resol(nside)
    coords = xr.Dataset(
        {
            "latitude": (["cells"], lat, {"units": "deg"}),
            "longitude": (["cells"], lon, {"units": "deg"}),
            "cell_ids": (["cells"], cell_ids),
        },
        coords={"resolution": ((), resolution, {"units": "rad"})},
    )

    return HealpyGridInfo(nside=nside, rot=rot, indices=indices, coords=coords)


def _compute_weights(source_lat, source_lon, *, nside, rot={"lat": 0, "lon": 0}):
    theta = (90.0 - (source_lat - rot["lat"])) / 180.0 * np.pi
    phi = -(source_lon - rot["lon"]) / 180.0 * np.pi

    new_dim = "neighbors"
    input_core_dims = unique(concat([source_lat.dims, source_lon.dims]))
    output_core_dims = [new_dim] + input_core_dims

    pix, weights = xr.apply_ufunc(
        partial(hp.get_interp_weights, nside, nest=True),
        theta,
        phi,
        input_core_dims=[input_core_dims, input_core_dims],
        output_core_dims=[output_core_dims, output_core_dims],
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {new_dim: 4}},
    )
    pix -= nside**2 * (np.min(pix) // nside**2)

    return pix, weights


def _weights_to_sparse(weights):
    rows = weights["dst_cell_ids"]
    cols = weights["src_cell_ids"]

    # TODO: reshape such that we get a matrix of (dst_cells, nj, ni)
    sizes = {"dst_cells": weights.attrs["n_dst_cells"]} | json.loads(
        weights.attrs["src_grid_dims"]
    )

    coo = sparse.COO(
        coords=np.stack([rows.data, cols.data]),
        data=weights.weights.data,
        fill_value=0,
        shape=(weights.attrs["n_dst_cells"], weights.attrs["n_src_cells"]),
    ).reshape(tuple(sizes.values()))

    sparse_weights = xr.DataArray(
        coo,
        dims=list(sizes),
        attrs=weights.attrs,
    )

    return sparse_weights


@dataclass(repr=False)
class HealpyRegridder:
    """Regrid a dataset to healpy face 0

    Parameters
    ----------
    input_grid : xarray.Dataset
        The input dataset. For now, it has to have the ``"latitude"`` and ``"longitude"`` coordinates.
    output_grid : HealpyGridInfo
        The target grid, containing healpix parameters like ``nside`` and ``rot``.
    """

    input_grid: xr.Dataset
    output_grid: HealpyGridInfo

    weights_path: str | os.PathLike | None = None

    weights: xr.Dataset = field(init=False)

    def __post_init__(self):
        coords = ["latitude", "longitude"]

        stacked_dim = "src_grid_cells"

        src_grid = self.input_grid
        src_grid_dims = unique(
            concat(src_grid.variables[coord].dims for coord in coords)
        )
        src_grid_sizes = {name: src_grid.sizes[name] for name in src_grid_dims}

        stacked_src_grid = src_grid.stack({stacked_dim: src_grid_dims}).reset_index(
            stacked_dim
        )

        dst_grid = self.output_grid.to_xarray()
        dst_grid_dims = unique(concat(dst_grid.variables[name].dims for name in coords))
        dst_grid_sizes = {name: dst_grid.sizes[name] for name in dst_grid_dims}

        src_cell_ids = xr.DataArray(
            np.arange(stacked_src_grid.sizes[stacked_dim]), dims=stacked_dim
        )

        stacked_variables = [stacked_src_grid[coord] for coord in coords]
        pix_, weights_ = _compute_weights(
            *stacked_variables,
            nside=self.output_grid.nside,
            rot=self.output_grid.rot,
        )
        _, aligned = xr.broadcast(pix_, src_cell_ids)

        self.weights = (
            xr.Dataset(
                {"dst_cell_ids": pix_, "weights": weights_, "src_cell_ids": aligned}
            )
            .set_coords(["dst_cell_ids", "src_cell_ids"])
            .stack({"ncol": ["neighbors", "src_grid_cells"]})
            .reset_index("ncol")
            .drop_vars(coords)
            .merge(dst_grid, combine_attrs="drop_conflicts")
            .assign_attrs(
                {
                    "src_grid_dims": json.dumps(src_grid_sizes),
                    "dst_grid_dims": json.dumps(dst_grid_sizes),
                    "n_src_cells": stacked_src_grid.sizes[stacked_dim],
                    "n_dst_cells": self.output_grid.nside**2,
                }
            )
        )

    def regrid_ds(self, ds):
        """Regrid a dataset on the same grid as the input grid

        The regridding method is restricted to linear interpolation so far.

        Parameters
        ----------
        ds : xarray.Dataset
            The input dataset.

        Returns
        -------
        regridded : xarray.Dataset
            The regridded dataset
        """
        # based on https://github.com/pangeo-data/xESMF/issues/222#issuecomment-1524041837

        def _apply(da, weights, normalization):
            import opt_einsum

            # 🐵 🔧
            xr.core.duck_array_ops.einsum = opt_einsum.contract

            ans = xr.dot(
                # drop all input coords, as those would most likely be broadcast
                da.variable,
                weights,
                # This dimension will be "contracted"
                # or summmed over after multiplying by the weights
                dims=src_dims,
            )

            # 🐵 🔧 : restore back to original
            xr.core.duck_array_ops.einsum = np.einsum

            normalized = ans / normalization

            return normalized

        # construct the sparse weights matrix and pre-compute normalization factors
        weights = _weights_to_sparse(self.weights)

        src_dims = list(json.loads(weights.attrs["src_grid_dims"]))
        normalization = weights.sum(src_dims).as_numpy()

        # regrid only those variables with the source dims
        vars_with_src_dims = [
            name
            for name, array in ds.variables.items()
            if set(src_dims).issubset(array.dims) and name not in weights.coords
        ]
        regridded = ds[vars_with_src_dims].map(
            _apply,
            weights=weights.chunk(),
            normalization=normalization,
        )

        # reshape to a healpy 2d grid and assign coordinates
        coords = (
            self.output_grid.to_xarray().rename_dims({"cells": "dst_cells"}).chunk()
        )
        reshaped = (
            regridded.merge(coords, combine_attrs="drop_conflicts")
            .pipe(self.output_grid.unstructured_to_2d, dim="dst_cells")
            .set_coords(["latitude", "longitude", "cell_ids"])
        )

        # merge in other variables, but skip those that are already set
        to_drop = set(reshaped.variables) & set(ds.variables)
        merged = xr.merge(
            [ds.drop_vars(to_drop), reshaped],
            combine_attrs="drop_conflicts",
        ).drop_dims(src_dims)

        # crop all-missing rows and columns
        cropped = merged.dropna(dim="x", how="all", subset=["H0"]).dropna(
            dim="y", how="all", subset=["H0"]
        )

        return cropped


def buffer_points(
    cell_ids,
    positions,
    *,
    buffer_size,
    nside,
    sphere_radius=6371e3,
    factor=4,
    intersect=False,
):
    """Select the cells within a circular buffer around the given positions

    Parameters
    ----------
    cell_ids : xarray.DataArray
        The cell ids within the given grid.
    positions : xarray.DataArray
        The positions of the points in cartesian coordinates.
    buffer_size : float
        The size of the circular buffer.
    nside : int
        The resolution of the healpix grid.
    sphere_radius : float, default: 6371000
        The radius of the underlying sphere, used to convert ``radius`` to radians. By
        default, this is the standard earth's radius in meters.
    factor : int, default: 4
        The increased resolution for the buffer search.
    intersect : bool, default: False
        If ``False``, select all cells where the center is within the buffer. If ``True``,
        select cells which intersect the buffer.

    Returns
    -------
    masks : xarray.DataArray
        The masks for each position. The cells within the buffer are ``True``, every other
        cell is set to ``False``.

    See Also
    --------
    pangeo_fish.healpy.geographic_to_astronomic
    pangeo_fish.healpy.astronomic_to_cartesian
    """

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
        input_core_dims=[["x", "y"], ["cartesian"]],
        kwargs={
            "radius": radius_,
            "nside": nside,
            "factor": factor,
            "intersect": intersect,
        },
        output_core_dims=[["x", "y"]],
        vectorize=True,
    )

    return masks.assign_coords(cell_ids=cell_ids)
