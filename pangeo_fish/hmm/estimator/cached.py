import os
from dataclasses import asdict, dataclass, replace

import movingpandas as mpd
import numpy as np
import xarray as xr
import zarr
from tlz.functoolz import compose_left, curry, pipe
from tlz.itertoolz import first

from pangeo_fish import tracks, utils
from pangeo_fish.hmm.decode import mean_track, modal_track, viterbi, viterbi2
from pangeo_fish.hmm.filter import _backward_zarr, _forward_zarr


@dataclass
class CachedEstimator:
    """Estimator to train and predict gaussian random walk hidden markov models

    This estimator caches intermediate data to a zarr store, allowing it to compute tracks
    that wouldn't fit into memory otherwise, even on very big machines.

    Parameters
    ----------
    predictor_factory : callable
        Factory for the predictor class. It expects the parameter ("sigma") as a keyword
        argument and returns the predictor instance.
    sigma : float, optional
        The primary model parameter: the standard deviation of the distance
        per time unit traveled by the fish, in the same unit as the grid coordinates.
    cache : str or zarr.Store
        Zarr store to write intermediate results to.
    """

    predictor_factory: callable
    sigma: float | None = None

    cache: str | os.PathLike | zarr.storage.Store = None
    progress: bool = False

    def to_dict(self):
        exclude = {"cache", "progress", "predictor_factory"}

        return {k: v for k, v in asdict(self).items() if k not in exclude}

    def set_params(self, **params):
        """Set the parameters on a new instance

        Parameters
        ----------
        params : dict
            Mapping of parameter name to new value.

        """
        return replace(self, **params)

    def _score(self, X, *, cache, spatial_dims=None, temporal_dims=None, progress=None):
        if self.sigma is None:
            raise ValueError("unset sigma, cannot run the filter")

        if not isinstance(cache, zarr.storage.Store):
            raise ValueError("requires a zarr store for now")
        else:
            cache_store = cache

        if progress is None:
            progress = self.progress

        if spatial_dims is None:
            spatial_dims = utils._detect_spatial_dims(X)
        if temporal_dims is None:
            temporal_dims = utils._detect_temporal_dims(X)

        dims = temporal_dims + spatial_dims

        # write the dataset to disk
        X.transpose(*dims).to_zarr(
            cache_store, group="emission", mode="w", consolidated=True
        )
        group = zarr.group(cache_store, overwrite=False)
        if "forward" in group:
            del group["forward"]

        # propagate
        forward = _forward_zarr(
            group["emission"],
            group.create_group("forward"),
            predictor=self.predictor_factory(sigma=self.sigma),
            progress=progress,
        )

        # open and return the score
        with np.errstate(divide="ignore"):
            value = -np.log(forward["normalizations"]).sum()
            return value if not np.isnan(value) else np.inf

    def _forward_backward_algorithm(
        self, X, cache, *, spatial_dims=None, temporal_dims=None, progress=None
    ):
        if self.sigma is None:
            raise ValueError("unset sigma, cannot run the filter")

        if not isinstance(cache, zarr.storage.Store):
            raise ValueError("requires a zarr store for now")
        else:
            cache_store = cache

        if progress is None:
            progress = self.progress

        if spatial_dims is None:
            spatial_dims = utils._detect_spatial_dims(X)
        if temporal_dims is None:
            temporal_dims = utils._detect_temporal_dims(X)

        dims = temporal_dims + spatial_dims

        # write the dataset to disk
        X.transpose(*dims).to_zarr(
            cache_store, group="emission", mode="w", consolidated=True
        )

        # open the root group
        group = zarr.group(cache_store, overwrite=False)

        # propagate
        _forward_zarr(
            group["emission"],
            group.create_group("forward", overwrite=True),
            predictor=self.predictor_factory(sigma=self.sigma),
            progress=progress,
        )
        _backward_zarr(
            group["forward"],
            group.create_group("backward", overwrite=True),
            predictor=self.predictor_factory(sigma=self.sigma),
            progress=progress,
        )

        # open and return a dataset
        return xr.open_dataset(cache_store, engine="zarr", chunks={}, group="backward")

    def predict_proba(
        self, X, *, cache=None, spatial_dims=None, temporal_dims=None, progress=None
    ):
        """Predict the state probabilities

        This is done by applying the forward-backward algorithm to the data.

        Parameters
        ----------
        X : xarray.Dataset
            The emission probability maps. The dataset should contain these variables:

            - ``initial``, the initial probability map
            - ``pdf``, the emission probabilities
            - ``mask``, a mask to select ocean pixels

        cache : str, pathlib.Path or zarr.Store
            Path to the cache store. Used to compute the state probabilities with nearly
            constant memory usage.
        spatial_dims : list of hashable, optional
            The spatial dimensions of the dataset.
        temporal_dims : list of hashable, optional
            The temporal dimensions of the dataset.

        Returns
        -------
        state_probabilities : xarray.DataArray
            The computed state probabilities

        Notes
        -----
        The convolution implementation does not allow skipping nan values. Thus, we
        replace these with zeroes, apply the filter, and at the end revert back to nans.
        """
        if cache is None and self.cache is None:
            raise ValueError("need to provide the cache file")
        elif cache is None:
            cache = self.cache

        state = self._forward_backward_algorithm(
            X.fillna(0),
            cache=cache,
            spatial_dims=spatial_dims,
            temporal_dims=temporal_dims,
            progress=progress,
        )
        return state.where(X["mask"])

    def score(
        self, X, *, cache=None, spatial_dims=None, temporal_dims=None, progress=None
    ):
        """Score the fit of the selected model to the data

        Apply the forward-backward algorithm to the given data, then return the
        negative logarithm of the normalization factors.

        Parameters
        ----------
        X : xarray.Dataset
            The emission probability maps. The dataset should contain these variables:

            - ``pdf``, the emission probabilities
            - ``mask``, a mask to select ocean pixels
            - ``initial``, the initial probability map

        spatial_dims : list of hashable, optional
            The spatial dimensions of the dataset.
        temporal_dims : list of hashable, optional
            The temporal dimensions of the dataset.

        Returns
        -------
        score : float
            The score for the fit with the current parameters.
        """
        if cache is None and self.cache is None:
            raise ValueError("need to provide the cache file")
        elif cache is None:
            cache = self.cache

        return self._score(
            X.fillna(0),
            cache=cache,
            spatial_dims=spatial_dims,
            temporal_dims=temporal_dims,
            progress=progress,
        )

    def decode(
        self,
        X,
        states=None,
        *,
        mode="viterbi",
        spatial_dims=None,
        temporal_dims=None,
        progress=False,
        additional_quantities=["distance", "speed"],
    ):
        """Decode the state sequence from the selected model and the data

        Parameters
        ----------
        X : xarray.Dataset
            The emission probability maps. The dataset should contain these variables:

            - ``pdf``, the emission probabilities
            - ``mask``, a mask to select ocean pixels
            - ``initial``, the initial probability map
            - ``final``, the final probability map (optional)

        states : xarray.Dataset, optional
            The precomputed state probability maps. The dataset should contain these variables:

            - ``states``, the state probabilities

        mode : str or list of str, default: "viterbi"
            The decoding method. Can be one of:

            - ``"mean"``: use the centroid of the state probabilities as decoded state
            - ``"mode"``: use the maximum of the state probabilities as decoded state
            - ``"viterbi"``: use the viterbi algorithm to determine the most probable states

            If a list of methods is given, decode using all methods in sequence.
        additional_quantities : None or list of str, default: ["distance", "speed"]
            Additional quantities to compute from the decoded tracks. Use ``None`` or an
            empty list to not compute anything.

            Possible values are:

            - ``"distance"``: distance to the previous track point in ``[km]``
            - ``"speed"``: average speed for the movement from the previous to the current
                track point, in ``[km/h]``

        spatial_dims : list of hashable, optional
            The spatial dimensions of the dataset.
        temporal_dims : list of hashable, optional
            The temporal dimensions of the dataset.
        """

        def maybe_compute_states(data):
            X, states = data

            if states is None:
                return self.predict_proba(
                    X, spatial_dims=spatial_dims, temporal_dims=temporal_dims
                )

            return states

        decoders = {
            "mean": compose_left(maybe_compute_states, mean_track),
            "mode": compose_left(maybe_compute_states, modal_track),
            "viterbi": compose_left(first, curry(viterbi, sigma=self.sigma)),
            "viterbi2": compose_left(first, curry(viterbi2, sigma=self.sigma)),
        }

        if not isinstance(mode, list):
            modes = [mode]
        else:
            modes = mode

        if len(modes) == 0:
            raise ValueError("need at least one mode")

        wrong_modes = [mode for mode in modes if mode not in decoders]
        if wrong_modes:
            raise ValueError(
                f"unknown {'mode' if len(modes) == 1 else 'modes'}: "
                + (mode if len(modes) == 1 else ", ".join(repr(mode) for mode in modes))
                + "."
                + " Choose one of {{{', '.join(sorted(decoders))}}}."
            )

        def maybe_show_progress(modes):
            if not progress:
                return modes

            return utils.progress_status(modes)

        decoded = [
            pipe(
                [X, states],
                decoders.get(mode),
                lambda x: x.compute(),
                curry(tracks.to_trajectory, name=mode),
                curry(tracks.additional_quantities, quantities=additional_quantities),
            )
            for mode in maybe_show_progress(modes)
        ]

        if len(decoded) > 1:
            return mpd.TrajectoryCollection(decoded)

        return decoded[0]
