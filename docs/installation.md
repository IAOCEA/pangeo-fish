# Installation

This documentation covers the installation process of `pangeo-fish` in a virtual Python environment.

In the following, we expect Conda to be installed on your Linux-based system.

For Windows users, we recommend WSL (2.0).

For more information, please refer to this page: [pangeo on Windows](https://gitlab.ifremer.fr/diam/Pangeo-on-Windows).

## For local use

If you are only interested in using `pangeo-fish` locally (i.e, without accessing remote data nor HPC resources), then all is needed is to create you create a virtual environment and install the package as well as its depedencies.

To do so, clone the `pangeo-fish`'s repository and navigate to it:

```console
git clone https://github.com/IAOCEA/pangeo-fish.git
cd pangeo-fish
```

Then, create a conda/mamba environment with the following command:

```console
micromamba create -n pangeo-fish -f ci/requirements/environment.yaml
```

This will create your environment with all the required libraries to make `pangeo-fish` work.

Finally, install the package itself from either its repository:

```console
pip install -e .
```

... or from `pip`:

```console
pip install pangeo-fish
```

## For HPC use

This section details the additional steps to setup `pangeo-fish` for HPC use.

As such, we assume the reader has access to HPC resources such as [Datarmor](https://www.ifremer.fr/fr/infrastructures-de-recherche/le-supercalculateur-datarmor).

_Please ensure to have completed the installation introduced above first!_

Install the `dask-hpcconfig` package and set the environnement as a Jupyter kernel:

```console
pip install dask-hpcconfig
ipython kernel install --name "pangeo-fish" --user.
```

Besides, you can refer to [this documentation on how to use pangeo on HPC](https://gitlab.ifremer.fr/diam/pangeo_on_HPC).

All of those steps should create a Python environment able to run the different Jupyter notebooks available.
