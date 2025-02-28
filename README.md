# pangeo-fish

## Installation

Since the package has not been released yet, it has to be installed from source:

```sh
git clone https://github.com/iaocea/pangeo-fish.git
cd pangeo-fish
```

The main dependencies are:

- xarray
- pandas
- numpy
- scipy
- numba
- more-itertools
- opt_einsum
- sparse
- healpy
- dask
- xdggs
- healpix-convolution

To avoid unexpected issues, we provide a file that specifies the different requirements for having `pangeo-fish` up and running.
Install the aforementioned `conda` environment with the following:

```sh
mamba env create -n pangeo-fish -f ci/requirements/environment.yaml
conda activate pangeo-fish
```

(use the drop-in replacement `mamba` or `micromamba` for faster results)

Finally, install the package itself with `pip`:

```sh
pip install -e .
```
