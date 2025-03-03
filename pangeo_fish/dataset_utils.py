import xarray as xr


def broadcast_variables(ds, variable_names):
    """Broadcast variables against each other

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to transform.
    variable_names : mapping of str to str
        The names of the variables to broadcast against each other. A
        mapping of variable names to new names.
    """
    old_names = list(variable_names.keys())
    new_names = list(variable_names.values())
    broadcasted = (
        ds[old_names]
        .reset_index(old_names)
        .rename_vars(variable_names)
        .reset_coords()
        .pipe(lambda ds: xr.broadcast(ds)[0])
        .set_coords(new_names)
    )

    return ds.merge(broadcasted)
