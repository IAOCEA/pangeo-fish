import itertools

import dask
import dask.array as da
import numba
import numpy as np
import xarray as xr

from pangeo_fish.distributions import gaussian_kernel


def mean_track(X, coords=["latitude", "longitude"]):
    probabilities = X["states"]
    dims = list(
        set(itertools.chain.from_iterable(probabilities[name].dims for name in coords))
    )
    grid = probabilities.reset_coords(coords).get(coords)
    return grid.weighted(probabilities).mean(dim=dims)


def modal_track(X, coords=["latitude", "longitude"]):
    probabilities = X["states"].drop_vars("cell_ids")
    dims = list(
        set(itertools.chain.from_iterable(probabilities[name].dims for name in coords))
    )
    (indices,) = dask.compute(probabilities.argmax(dim=dims))

    return xr.Dataset(
        {name: probabilities[name][indices] for name in coords}
    ).reset_coords(coords)


@numba.njit
def kernel_overlap(pos, shape, kernel_shape):
    px, py = pos
    xmax, ymax = shape
    kx, ky = kernel_shape

    a_xmin = max(px - kx // 2, 0)
    a_xmax = min(px + kx // 2 + 1, xmax)
    a_ymin = max(py - ky // 2, 0)
    a_ymax = min(py + ky // 2 + 1, ymax)

    k_xmin = max(0 - px + kx // 2, 0)
    k_xmax = min(xmax - px + kx // 2, kx)
    k_ymin = max(0 - py + ky // 2, 0)
    k_ymax = min(ymax - py + ky // 2, kx)

    return ((a_xmin, a_xmax), (a_ymin, a_ymax)), ((k_xmin, k_xmax), (k_ymin, k_ymax))


@numba.njit
def kernel_state_metric(previous_state_metric, previous_positions, pdf, kernel):
    possible_positions = previous_state_metric != -np.inf

    # output arrays
    state_metric = np.full_like(previous_state_metric, fill_value=-np.inf)
    positions = np.full_like(previous_positions, fill_value=-1)

    # actual processing
    nx, ny = pdf.shape
    for x in range(nx):
        for y in range(ny):
            # skip if unreachable
            if not possible_positions[x, y]:
                continue

            # compute overlap
            (ax, ay), (kx, ky) = kernel_overlap((x, y), pdf.shape, kernel.shape)

            # current branch metric
            branch_metric = (
                pdf[ax[0] : ax[1], ay[0] : ay[1]] + kernel[kx[0] : kx[1], ky[0] : ky[1]]
            )

            # current state metric
            new_state_metric = branch_metric + previous_state_metric[x, y]

            # write the result
            region = state_metric[ax[0] : ax[1], ay[0] : ay[1]]

            mask = new_state_metric > region

            state_metric[ax[0] : ax[1], ay[0] : ay[1]] = np.where(
                mask, new_state_metric, region
            )
            positions[ax[0] : ax[1], ay[0] : ay[1]] = np.where(
                mask, x * ny + y, positions[ax[0] : ax[1], ay[0] : ay[1]]
            )

    return state_metric, positions


def viterbi(emission, sigma):
    """
    Parameters
    ----------
    emission : xarray.Dataset
        The emission probability dataset containing land mask and initial
        probabilities and the pdf.
    sigma : float
        The coefficient of diffusion in pixel

    Returns
    -------
    state_metrics : xarray.DataArray
        The evolution of state metrics over time
    positions : xarray.DataArray
        The integer positions of the tracks
        TODO: translate this to label space without computing
    """

    def apply_kernel_state_metric(
        pdf, previous_state_metric, previous_branches, kernel
    ):
        if isinstance(pdf, da.Array):
            return da.apply_gufunc(
                kernel_state_metric,
                "(x,y),(x,y),(x,y),(n,m)->(x,y),(x,y)",
                previous_state_metric,
                previous_branches,
                pdf,
                kernel,
                output_dtypes=[float, int],
            )
        else:
            return kernel_state_metric(
                previous_state_metric, previous_branches, pdf, kernel
            )

    def decode_most_probable_track(pdf, sigma, ocean_mask):
        kernel = gaussian_kernel(np.full(shape=(2,), fill_value=sigma))

        [pos0] = dask.compute(np.argmax(pdf[0, ...]))

        # all in log space
        state_metrics_ = [pdf[0, ...]]

        initial_branches = np.where(pdf[0, ...] != -np.inf, pos0, -1)
        branches_ = [initial_branches]

        for index in range(1, pdf.shape[0]):
            state_metric, positions = apply_kernel_state_metric(
                pdf[index, ...],
                state_metrics_[index - 1],
                branches_[index - 1],
                np.log(kernel),
            )

            state_metrics_.append(np.where(ocean_mask, state_metric, -np.inf))
            branches_.append(np.where(ocean_mask, positions, -1))

        state_metrics = np.stack(state_metrics_, axis=0)
        branches = np.stack(branches_, axis=0)

        reshaped_branches = branches.reshape(branches.shape[0], -1)

        most_likely_position = np.argmax(state_metrics[-1, ...])

        positions = [most_likely_position]
        for index in range(1, state_metrics.shape[0]):
            branch_index = reshaped_branches.shape[0] - index
            positions.append(reshaped_branches[branch_index, positions[index - 1]])

        return state_metrics, np.stack(positions[::-1])

    pdf = emission["pdf"].copy()
    pdf[{"time": 0}] = emission.initial

    state_metrics, positions = xr.apply_ufunc(
        decode_most_probable_track,
        np.log(pdf.fillna(0)),
        sigma,
        emission.mask,
        input_core_dims=[("x", "y"), (), ("x", "y")],
        output_core_dims=[("x", "y"), ()],
        dask="allowed",
    )

    coords = emission[["longitude", "latitude"]].stack(z=["x", "y"])
    track = coords.isel(z=positions, drop=True)

    return track.drop_vars(["cell_ids", "z", "x", "y"])


@numba.njit
def _propagate_timestep(M, kernel, emission, Tprevx, Tprevy):
    row, col = M.shape
    ks = kernel.shape[0]

    Mtemp = np.full((row, col), fill_value=-np.inf)
    Ttempx = np.full((row, col), fill_value=-1, dtype="int16")
    Ttempy = np.full((row, col), fill_value=-1, dtype="int16")

    for x in range(col):
        for y in range(row):
            if M[y, x] == -np.inf:
                continue

            kminlat = max(ks // 2 - y, 0)
            kmaxlat = min((ks - 1) - (y + ks // 2 - (row - 1)), ks - 1)
            kminlong = max(ks // 2 - x, 0)
            kmaxlong = min((ks - 1) - (x + ks // 2 - (col - 1)), ks - 1)

            mminlat = max(y - ks // 2, 0)
            mmaxlat = min(y + ks // 2, row - 1)
            mminlong = max(x - ks // 2, 0)
            mmaxlong = min(x + ks // 2, col - 1)

            B = (
                emission[mminlat : mmaxlat + 1, mminlong : mmaxlong + 1]
                + kernel[kminlat : kmaxlat + 1, kminlong : kmaxlong + 1]
            )

            Msub = B + M[y, x]

            Mupdate = Mtemp[mminlat : mmaxlat + 1, mminlong : mmaxlong + 1]
            Txupdate = Ttempx[mminlat : mmaxlat + 1, mminlong : mmaxlong + 1]
            Tyupdate = Ttempy[mminlat : mmaxlat + 1, mminlong : mmaxlong + 1]

            update = Msub > Mupdate

            Mtemp[mminlat : mmaxlat + 1, mminlong : mmaxlong + 1] = np.where(
                update, Msub, Mupdate
            )
            Ttempx[mminlat : mmaxlat + 1, mminlong : mmaxlong + 1] = np.where(
                update, x, Txupdate
            )
            Ttempy[mminlat : mmaxlat + 1, mminlong : mmaxlong + 1] = np.where(
                update, y, Tyupdate
            )

    return Mtemp, Ttempx, Ttempy


@numba.njit
def _reorder_track(Tprevx, Tprevy, Ttempx, Ttempy, index, M):
    row, col = M.shape

    Tx = np.full_like(Tprevx, fill_value=-1)
    Ty = np.full_like(Tprevy, fill_value=-1)

    for x in range(col):
        for y in range(row):
            if M[y, x] == -np.inf:
                continue

            Tx[y, x, :index] = Tprevx[Ttempy[y, x], Ttempx[y, x], :index]
            Ty[y, x, :index] = Tprevy[Ttempy[y, x], Ttempx[y, x], :index]
            Tx[y, x, index] = x
            Ty[y, x, index] = y

    return Tx, Ty


def _viterbi(emission, land_mask, sigma):
    kernel = np.log(gaussian_kernel(np.array([sigma, sigma]), type="discrete"))
    emission_ = np.log(emission)

    M = dask.compute(emission_[0, ...])[0]

    pos0 = np.argmax(M)
    y0, x0 = np.unravel_index(pos0, M.shape)

    Tprevx = np.full(M.shape + emission_.shape[:1], fill_value=-1, dtype="int16")
    Tprevx[y0, x0, 0] = x0
    Tprevy = np.full(M.shape + emission_.shape[:1], fill_value=-1, dtype="int16")
    Tprevy[y0, x0, 0] = y0

    for index in range(1, emission_.shape[0]):
        pdf = dask.compute(emission_[index, ...])[0]

        Mtemp, Ttempx, Ttempy = _propagate_timestep(M, kernel, pdf, Tprevx, Tprevy)
        Mtemp[land_mask] = -np.inf

        Tx, Ty = _reorder_track(Tprevx, Tprevy, Ttempx, Ttempy, index, Mtemp)

        M = Mtemp
        Tprevx = Tx
        Tprevy = Ty

    reshaped_M = np.reshape(M, -1)
    reshaped_x = np.reshape(Tprevx, (-1, emission.shape[0]))
    reshaped_y = np.reshape(Tprevy, (-1, emission.shape[0]))

    pos = np.argmax(reshaped_M)
    y, x = reshaped_y[pos, :], reshaped_x[pos, :]

    return y, x


def viterbi2(emission, sigma):
    pdf = emission["pdf"].copy()
    pdf[{"time": 0}] = emission.initial

    # different order for x and y, so we need to swap it
    x, y = xr.apply_ufunc(
        _viterbi,
        pdf.fillna(0),
        np.logical_not(emission.mask),  # ocean mask → land mask
        input_core_dims=[("x", "y"), ("x", "y")],
        output_core_dims=[(), ()],
        kwargs={"sigma": sigma},
        dask="allowed",
    )

    return (
        emission[["time", "latitude", "longitude"]].isel(x=x, y=y).drop_vars("cell_ids")
    )
