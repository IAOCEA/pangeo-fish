from dataclasses import asdict, dataclass, field, replace

import movingpandas as mpd
import numpy as np
from tlz.functoolz import compose_left, curry, pipe
from tlz.itertoolz import first

from pangeo_fish import tracks, utils
from pangeo_fish.hmm.decode import mean_track, modal_track, viterbi, viterbi2
from pangeo_fish.hmm.filter import forward, forward_backward, score
from pangeo_fish.hmm.prediction import Predictor


@dataclass
class EagerEstimator:
    """Estimator to train and predict gaussian random walk hidden markov models

    This estimator performs all calculations eagerly and assumes all data can fit into memory.

    Parameters
    ----------
    predictor_factory : callable
        Factory for the predictor class. It expects the parameter ("sigma") as a keyword
        argument and returns the predictor instance.
    sigma : float, optional
        The primary model parameter: the standard deviation of the distance
        per time unit traveled by the fish, in the same unit as the grid coordinates.
    """

    predictor_factory: callable
    sigma: float | None = None

    predictor: Predictor | None = field(default=None, init=False)

    def to_dict(self):
        exclude = {"predictor_factory"}

        return {k: v for k, v in asdict(self).items() if k not in exclude}

    def set_params(self, **params):
        """Set the parameters on a new instance

        Parameters
        ----------
        **params
            Mapping of parameter name to new value.
        """
        return replace(self, **params)

    def _score(self, X, *, spatial_dims=None, temporal_dims=None):
        if self.sigma is None:
            raise ValueError("unset sigma, cannot run the filter")

        if spatial_dims is None:
            spatial_dims = utils._detect_spatial_dims(X)
        if temporal_dims is None:
            temporal_dims = utils._detect_temporal_dims(X)

        dims = temporal_dims + spatial_dims

        X_ = X.transpose(*dims)

        predictor: Predictor
        if self.predictor is None:
            predictor = self.predictor_factory(sigma=self.sigma)
            self.predictor = predictor
        else:
            predictor = self.predictor

        value = score(
            emission=X_["pdf"].data,
            mask=X_["mask"].data,
            initial_probability=X_["initial"].data,
            predictor=predictor,
        )

        return value if not np.isnan(value) else np.inf

    def _forward_algorithm(self, X, *, spatial_dims=None, temporal_dims=None):
        if self.sigma is None:
            raise ValueError("unset sigma, cannot run the filter")

        if spatial_dims is None:
            spatial_dims = utils._detect_spatial_dims(X)
        if temporal_dims is None:
            temporal_dims = utils._detect_temporal_dims(X)

        predictor = self.predictor_factory(sigma=self.sigma)
        filtered = forward(
            emission=X["pdf"].data,
            mask=X["mask"].data,
            initial_probability=X["initial"].data,
            predictor=predictor,
        )
        return X["pdf"].copy(data=filtered)

    def _forward_backward_algorithm(self, X, *, spatial_dims=None, temporal_dims=None):
        if self.sigma is None:
            raise ValueError("unset sigma, cannot run the filter")

        if spatial_dims is None:
            spatial_dims = utils._detect_spatial_dims(X)
        if temporal_dims is None:
            temporal_dims = utils._detect_temporal_dims(X)

        dims = temporal_dims + spatial_dims
        X_ = X.transpose(*dims)

        predictor = self.predictor_factory(sigma=self.sigma)
        filtered = forward_backward(
            emission=X_["pdf"].data,
            mask=X_["mask"].data,
            initial_probability=X_["initial"].data,
            predictor=predictor,
        )

        return X["pdf"].copy(data=filtered)

    def predict_proba(self, X, *, spatial_dims=None, temporal_dims=None):
        """Predict the state probabilities

        This is done by applying the forward-backward algorithm to the data.

        Parameters
        ----------
        X : xarray.Dataset
            The emission probability maps. The dataset should contain these variables:

            - ``initial``, the initial probability map
            - ``pdf``, the emission probabilities
            - ``mask``, a mask to select ocean pixels

            Due to the convolution method we use today, we can't pass np.nan, thus we send ``x.fillna(0)``, but drop the values whihch are less than 0 and put them back to np.nan when we return the value.
        spatial_dims : list of hashable, optional
            The spatial dimensions of the dataset.
        temporal_dims : list of hashable, optional
            The temporal dimensions of the dataset.

        Returns
        -------
        state_probabilities : xarray.DataArray
            The computed state probabilities
        """
        state = self._forward_backward_algorithm(
            X.fillna(0),
            spatial_dims=spatial_dims,
            temporal_dims=temporal_dims,
        )
        state = state.where(state > 0)
        return state.rename("states")

    def score(self, X, *, spatial_dims=None, temporal_dims=None):
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
        return self._score(
            X.fillna(0), spatial_dims=spatial_dims, temporal_dims=temporal_dims
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

            - ``pdf``: the emission probabilities
            - ``mask``: a mask to select ocean pixels
            - ``initial``: the initial probability map
            - ``final``: the final probability map (optional)

        states : xarray.Dataset, optional
            The precomputed state probability maps. The dataset should contain these variables:

            - ``states``: the state probabilities

        mode : str or list of str, default: "viterbi"
            The decoding method. Possible values are:

            - ``"mean"``: use the centroid of the state probabilities as decoded state.
            - ``"mode"``: use the maximum of the state probabilities as decoded state.
            - ``"viterbi"``: use the Viterbi algorithm to determine the most probable states.

            If a list of methods is given, decode using all methods in sequence.

        additional_quantities : None or list of str, default: ["distance", "speed"]
            Additional quantities to compute from the decoded tracks. Use ``None`` or an
            empty list to compute nothing.

            Possible values are:

            - ``"distance"``: distance to the previous track point in ``[km]``.
            - ``"speed"``: average speed from the previous to the current track point in ``[km/h]``.

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
