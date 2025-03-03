API Reference
=============


.. High-level Functions
.. --------------------


.. .. currentmodule:: pangeo_fish.helpers
.. .. autosummary::
..    :toctree: generated

   .. to_healpix
   .. regrid_to_2d
   .. load_tag
   .. update_stations
   .. plot_tag
   .. load_model
   .. compute_diff
   .. open_diff_dataset
   .. regrid_dataset
   .. compute_emission_pdf
   .. compute_acoustic_pdf
   .. combine_pdfs
   .. optimize_pdf
   .. predict_positions
   .. plot_trajectories
   .. open_distributions
   .. plot_distributions
   .. render_frames
   .. render_distributions


I/O
---

.. currentmodule:: pangeo_fish.io
.. autosummary::
   :toctree: generated

   open_tag
   open_copernicus_catalog
   open_copernicus_zarr
   prepare_dataset
   save_trajectories
   read_trajectories
   save_html_hvplot
   tz_convert


Re-gridding
-----------

.. currentmodule:: pangeo_fish
.. autosummary::
   :toctree: generated

   grid.center_longitude
   healpy.geographic_to_astronomic
   healpy.astronomic_to_cartesian
   healpy.astronomic_to_cell_ids
   healpy.buffer_points


Emission Computations
---------------------

.. currentmodule:: pangeo_fish
.. autosummary::
   :toctree: generated

   acoustic.emission_probability
   pdf.normal
   pdf.combine_emission_pdf
   diff.diff_z


Tag/time Operations
-------------------

.. currentmodule:: pangeo_fish
.. autosummary::
   :toctree: generated

   tags.to_time_slice
   tags.adapt_model_time
   tags.reshape_by_bins
   cf.bounds_to_bins
   dataset_utils.broadcast_variables


Visualization
-------------

.. currentmodule:: pangeo_fish.visualization
.. autosummary::
   :toctree: generated

   create_single_frame
   render_frame
   plot_map
   plot_trajectories
   filter_by_states


Estimators
----------

.. currentmodule:: pangeo_fish.hmm.estimator
.. autosummary::
   :toctree: generated

   EagerEstimator
   CachedEstimator


Predictors
----------

.. currentmodule:: pangeo_fish.hmm.prediction
.. autosummary::
   :toctree: generated

   Gaussian2DCartesian
   Gaussian1DHealpix


Searches
--------

.. currentmodule:: pangeo_fish.hmm.optimize.scipy
.. autosummary::
   :toctree: generated

   GridSearch
   EagerBoundsSearch
   TargetBoundsSearch


Low-level Functions
-------------------

Distributions
#############

.. currentmodule:: pangeo_fish.distributions
.. autosummary::
   :toctree: generated

   planar2d.create_covariances
   planar2d.normal_at
   healpix.normal_at


Trajectory Generation
#####################

.. currentmodule:: pangeo_fish.hmm.decode
.. autosummary::
   :toctree: generated

   mean_track
   modal_track
   viterbi


HMM Filtering
#############

.. currentmodule:: pangeo_fish.hmm.filter
.. autosummary::
   :toctree: generated

   score
   forward
   backward
   forward_backward
