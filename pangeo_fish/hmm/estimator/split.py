import xarray as xr

from ... import utils
from ..filter import forward, score


class EagerScoreEstimator:
    """Estimator to train and predict gaussian random walk hidden markov models

    Parameters
    ----------
    sigma : float, default: None
        The primary model parameter: the standard deviation of the distance
        per time unit traveled by the fish, in the same unit as the grid coordinates.
    truncate : float, default: 4.0
        The cut-off limit of the filter. This can be used, together with `sigma`, to
        calculate the maximum distance per time unit traveled by the fish.
    """

    def __init__(self, *, sigma=None, truncate=4.0):
        self.sigma = sigma
        self.truncate = truncate

    def set_params(self, **params):
        """set the parameters on a new instance

        Parameters
        ----------
        **params
            Mapping of parameter name to new value.
        """
        old_params = {"sigma": self.sigma, "truncate": self.truncate}
        new_params = old_params | params

        return type(self)(**new_params)

    def _score(self, X, *, spatial_dims=None, temporal_dims=None):
        def _algorithm(emission, mask, initial, final, *, sigma, truncate):
            return score(
                emission,
                mask=mask,
                initial_probability=initial,
                final_probability=final,
                sigma=sigma,
                truncate=truncate,
            )

        if spatial_dims is None:
            spatial_dims = utils._detect_spatial_dims(X)
        if temporal_dims is None:
            temporal_dims = utils._detect_temporal_dims(X)

        if "final" in X:
            final = X.final
            final_dims = spatial_dims
        else:
            final = None
            final_dims = ()

        input_core_dims = [
            temporal_dims + spatial_dims,
            spatial_dims,
            spatial_dims,
            final_dims,
        ]

        return xr.apply_ufunc(
            _algorithm,
            X.pdf.fillna(0),
            X.mask,
            X.initial,
            final,
            kwargs={"sigma": self.sigma, "truncate": self.truncate},
            input_core_dims=input_core_dims,
            output_core_dims=[()],
            dask="allowed",
        )

    def _forward_algorithm(self, X, *, spatial_dims=None, temporal_dims=None):
        def _algorithm(emission, mask, initial, final, *, sigma, truncate):
            return forward(
                emission=emission,
                mask=mask,
                initial_probability=initial,
                final_probability=final,
                sigma=sigma,
                truncate=truncate,
            )

        if spatial_dims is None:
            spatial_dims = utils._detect_spatial_dims(X)
        if temporal_dims is None:
            temporal_dims = utils._detect_temporal_dims(X)

        if "final" in X:
            final = X.final
            final_dims = spatial_dims
        else:
            final = None
            final_dims = ()

        input_core_dims = [
            temporal_dims + spatial_dims,
            spatial_dims,
            spatial_dims,
            final_dims,
        ]

        return xr.apply_ufunc(
            _algorithm,
            X.pdf,
            X.mask,
            X.initial,
            final,
            kwargs={"sigma": self.sigma, "truncate": self.truncate},
            input_core_dims=input_core_dims,
            output_core_dims=[
                temporal_dims,
                temporal_dims + spatial_dims,
            ],
            dask="allowed",
        )

    def _forward_backward_algorithm(self, X, *, spatial_dims=None, temporal_dims=None):
        def _algorithm(emission, mask, initial, final, *, sigma, truncate):
            forward_state = forward(
                emission=emission,
                sigma=sigma,
                mask=mask,
                initial_probability=initial,
                final_probability=final,
                truncate=truncate,
            )
            backward_state = forward(
                emission=forward_state[::-1, ...],
                sigma=sigma,
                mask=mask,
                initial_probability=forward_state[-1, ...],
                final_probability=forward_state[0, ...],
                truncate=truncate,
            )

            return backward_state[::-1, ...]

        if spatial_dims is None:
            spatial_dims = utils._detect_spatial_dims(X)
        if temporal_dims is None:
            temporal_dims = utils._detect_temporal_dims(X)

        if "final" in X:
            final = X.final
            final_dims = spatial_dims
        else:
            final = None
            final_dims = ()

        input_core_dims = [
            temporal_dims + spatial_dims,
            spatial_dims,
            spatial_dims,
            final_dims,
        ]

        return xr.apply_ufunc(
            _algorithm,
            X.pdf,
            X.mask,
            X.initial,
            final,
            kwargs={"sigma": self.sigma, "truncate": self.truncate},
            input_core_dims=input_core_dims,
            output_core_dims=[temporal_dims + spatial_dims],
            dask="allowed",
        )

    def predict_proba(self, X, *, spatial_dims=None, temporal_dims=None):
        """predict the state probabilities

        This is done by applying the forward-backward algorithm to the data.

        Parameters
        ----------
        X : Dataset
            The emission probability maps. The dataset should contain these variables:
            - `initial`, the initial probability map
            - `final`, the final probability map
            - `pdf`, the emission probabilities
            - `mask`, a mask to select ocean pixels
        spatial_dims : list of hashable, optional
            The spatial dimensions of the dataset.
        temporal_dims : list of hashable, optional
            The temporal dimensions of the dataset.

        Returns
        -------
        state_probabilities : DataArray
            The computed state probabilities
        """
        state = self._forward_backward_algorithm(
            X, spatial_dims=spatial_dims, temporal_dims=temporal_dims
        )
        return state

    def score(self, X, *, spatial_dims=None, temporal_dims=None):
        """score the fit of the selected model to the data

        Apply the forward-backward algorithm to the given data, then return the
        negative logarithm of the normalization factors.

        Parameters
        ----------
        X : Dataset
            The emission probability maps. The dataset should contain these variables:
            - `pdf`, the emission probabilities
            - `mask`, a mask to select ocean pixels
            - `initial`, the initial probability map
            - `final`, the final probability map (optional)
        spatial_dims : list of hashable, optional
            The spatial dimensions of the dataset.
        temporal_dims : list of hashable, optional
            The temporal dimensions of the dataset.

        Return
        ------
        score : float
            The score for the fit with the current parameters.
        """
        return self._score(X, spatial_dims=spatial_dims, temporal_dims=temporal_dims)
