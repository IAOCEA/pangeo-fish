"""Implements operations for merging probability distributions."""

import cf_xarray  # noqa: F401
import more_itertools
import scipy.stats
import xarray as xr
from more_itertools import first

from pangeo_fish.utils import _detect_spatial_dims, normalize


# also try: multivariate_normal, gaussian_kde
# TODO: use numba to vectorize (using `guvectorize`?)
def normal(samples, mean, std, *, dims):
    """compute the combined pdf of independent layers

    Parameters
    ----------
    samples : DataArray, Variable, or array-like
        The samples to compute the pdf from
    mean : float
        The mean of the distribution
    std : float
        The standard deviation of the distribution
    dims : list of hashable
        The dimension to compute the pdf along

    Returns
    -------
    pdf : DataArray
        The computed pdf
    """

    def _pdf(samples, mean, std):
        return scipy.stats.norm.pdf(samples, mean, std)

    if isinstance(std, int | float) or std.size == 1:
        param_dims = []
    else:
        param_dims = mean.dims

    result = xr.apply_ufunc(
        _pdf,
        samples,
        mean,
        std**2,
        dask="parallelized",
        input_core_dims=[dims, param_dims, param_dims],
        output_core_dims=[dims],
        exclude_dims=set(param_dims),
        vectorize=True,
    )
    return result.rename("pdf").drop_attrs(deep=False)


def combine_emission_pdf(raw, exclude=("initial", "final", "mask")):
    exclude = [n for n in more_itertools.always_iterable(exclude) if n in raw.variables]

    to_combine = [name for name in raw.data_vars if name not in exclude]
    if len(to_combine) == 1:
        pdf = raw[first(to_combine)].rename("pdf")
    else:
        pdf = (
            raw[to_combine]
            .to_array(dim="pdf")
            .prod(dim="pdf", skipna=False)
            .rename("pdf")
        )

    if "final" in raw:
        pdf[{"time": -1}] = pdf[{"time": -1}] * raw["final"]

    spatial_dims = _detect_spatial_dims(raw)
    return xr.merge([raw[exclude], pdf.pipe(normalize, spatial_dims)])
