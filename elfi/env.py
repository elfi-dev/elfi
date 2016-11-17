from distributed import Client, LocalCluster
import socket


from collections import defaultdict


_globals = defaultdict(lambda: None)
_whitelist = ['client', 'inference_tasks']


def set(**kwargs):
    """Set global environment settings for ELFI

    Parameters
    ----------
    client : dask.distributed client object

    Examples
    --------
    c = Client()
    set(client = c)

    """
    for k in kwargs:
        if k not in _whitelist:
            raise ValueError('Unrecognized ELFI environment setting %s' % k)
        _globals[k] = kwargs[k]


def get(key):
    if key not in _whitelist:
        raise ValueError('Unrecognized ELFI environment setting %s' % key)
    return _globals[key]


def client(n_workers=None, threads_per_worker=None):
    """Gets the current framework client, or constructs a local one using the
    parameters if none is found.

    Parameters
    ----------
    n_workers : int
    threads_per_worker : int

    Returns
    -------
    `distributed.Client`
    """

    """"""

    c = get("client")
    if c is None or c.status != 'running':
        cluster_kwargs = {"n_workers": n_workers,
                          "threads_per_worker": threads_per_worker,
                          # Do not start the diagnostics server (bokeh) by default
                          "diagnostics_port": None
                          }
        try:
            cluster = LocalCluster(**cluster_kwargs)
        except (OSError, socket.error):
            # Try with a random port
            cluster_kwargs["scheduler_port"] = 0
            cluster = LocalCluster(**cluster_kwargs)
        c = Client(cluster, set_as_default=True)
        set(client = c)
    return c


def inference_task(key='default', default_class=None):
    """Provides the current `InferenceTask`. If default class is provided
    creates a new instance with a given key if none is found.

    Parameters
    ----------
    key : hashable
    default_class : callable (optional)

    Returns
    -------
    `InferenceTask`

    """
    itasks = get("inference_tasks")
    if itasks is None:
        itasks = {}
        set(inference_tasks = itasks)

    if key not in itasks:
        if default_class is None:
            raise IndexError("Could not find an inference task with a key {}".format(key))
        itasks[key] = default_class()

    return itasks[key]