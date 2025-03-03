import re

scheduler_address_re = re.compile(r"tcp://\[?[0-9a-f:.]+\]?:[0-9]+")
scheme_re = re.compile(r"(?P<type>[^:]+):(?P<data>.+)")


def create_cluster(spec, *, scale=None, additional_kwargs=None):
    """Create a cluster from a spec and return a client

    Parameters
    ----------
    spec : str
        The method of creating a cluster:

        - "local": local cluster
        - the scheduler address of a running cluster: ``tcp://<ip>:<port>``
        - ``dask-jobqueue:<path-to-spec-file>``
        - ``dask-hpcconfig:<name>``

    scale : int or mapping, optional
        Scale the cluster after creation. Not allowed with scheduler addresses.
    additional_kwargs : mapping, optional
        Additional keyword arguments passed to the cluster
    """
    if additional_kwargs is None:
        additional_kwargs = {}

    if spec == "local":
        from distributed import LocalCluster

        cluster = LocalCluster(**additional_kwargs)
    elif scheduler_address_re.match(spec) is not None:
        from distributed import Client

        client = Client(spec, **additional_kwargs)

        if scale is not None:
            raise ValueError(
                "it is impossible to modify the cluster using the scheduler address"
            )

        return client
    else:
        match = scheme_re.match(spec)
        if match is None:
            raise ValueError(f"invalid scheme format: {spec!r}")

        groups = match.groupdict()
        if groups["type"] == "dask-jobqueue":
            raise NotImplementedError("does not work yet")
        elif groups["type"] == "dask-hpcconfig":
            import dask_hpcconfig

            cluster = dask_hpcconfig.cluster(groups["data"], **additional_kwargs)
        else:
            raise ValueError("unknown cluster scheme")

    if scale is not None:
        if isinstance(scale, dict):
            cluster.scale(**scale)
        else:
            cluster.scale(scale)

    client = cluster.get_client()
    client.wait_for_workers(1)

    return client
