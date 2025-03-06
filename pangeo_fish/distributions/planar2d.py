import more_itertools
import numpy as np
import xarray as xr
from scipy.special import ive
from scipy.stats import multivariate_normal


def create_covariances(cov, coord_names):
    if isinstance(cov, int | float) or cov.size == 1:
        # only for 2D, so we have to repeat
        cov_ = cov * np.eye(2)
    elif cov.size == 2:
        cov_ = np.eye(2) @ cov
    else:
        cov_ = cov

    return xr.DataArray(
        cov_, dims=["i", "j"], coords={"i": coord_names, "j": coord_names}
    )


def normal_at(grid, *, pos, cov, axes=["X", "Y"], normalize=False):
    """multivariate normal distribution

    Parameters
    ----------
    grid : Dataset
        The reference grid.
    pos : Dataset
        The position of the mean of the distribution
    cov : DataArray
        The covariance matrix of the distribution. Has to have a `i` dimension
        for variances and `i` and `j` for covariances. Both need to have `axes`
        as coordinate values.
    axes : list of hashable, default: ["X", "Y"]
        The coordinates to use. Can be anything that `cf-xarray`'s `.cf` accessor understands.
    normalize : bool, default: False
        Normalize the distribution before returning.
    """
    pos_ = tuple(pos.cf[axis].item() for axis in axes)

    ndims = {grid.cf[axis].name: grid.cf[axis].ndim for axis in axes}
    unique_ndims = set(ndims.values())

    grid_coords = grid.cf[axes]

    if "bounds" in grid_coords.dims:
        coords_ = grid_coords.drop_dims("bounds")
    else:
        coords_ = grid_coords

    if len(unique_ndims) != 1:
        raise ValueError("coordinates need to have the same number of dimensions")
    elif unique_ndims == {1}:
        variables = coords_.reset_index(
            [name for name in coords_.variables if name in coords_.dims]
        )
        renamed = variables.rename_vars(
            {name: f"{name}_coord" for name in coords_.coords if name in coords_.dims}
        )
        (coords,) = xr.broadcast(renamed.reset_coords(renamed.coords))
    else:
        # non-dimension coordinates
        coord_names = [
            coord for coord in coords_.coords.keys() if coord not in coords_.dims
        ]
        coords = coords_.reset_coords(coord_names)

    variable_name = "normal_pdf"
    input_grid = (
        coords.to_array(dim="axes")
        .to_dataset(name=variable_name)
        .compute()
        .assign_coords(grid_coords.coords)
    )
    cov_ = cov.sortby([xr.DataArray(axes, dims=d) for d in cov.dims]).data

    distribution = multivariate_normal(mean=pos_, cov=cov_)

    pdf = xr.apply_ufunc(
        distribution.pdf,
        input_grid,
        input_core_dims=[list(coords.dims) + ["axes"]],
        output_core_dims=[list(coords.dims)],
        keep_attrs=False,
    ).assign_coords(grid_coords)
    if normalize:
        pdf = pdf / pdf.sum()

    return pdf[variable_name]


def zeros(coords, dtype=float):
    sizes = coords.sizes
    dims, shape = (tuple(_) for _ in more_itertools.unzip(sizes.items()))
    data = np.zeros(shape=shape, dtype=dtype)
    return xr.DataArray(data=data, dims=dims, coords=coords.coords)


def _discrete_gaussian_1d(sigma, radius):
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    return ive(x, sigma2)


def _continuous_gaussian_1d(sigma, radius):
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()

    return phi_x


def gaussian_kernel(sigma, truncate=4.0, type="continuous"):
    std = sigma.astype(float)
    lw = (truncate * std + 0.5).astype(int)

    generation_funcs = {
        "discrete": _discrete_gaussian_1d,
        "continuous": _continuous_gaussian_1d,
    }
    _gaussian_1d = generation_funcs.get(type)
    if _gaussian_1d is None:
        raise ValueError(
            "unknown type {type}, choose one of {sorted(generation_funcs.keys())}"
        )

    kernel = np.outer(_gaussian_1d(std[0], lw[0]), _gaussian_1d(std[1], lw[1]))
    return kernel / kernel.sum()
