from distributed import Client, LocalCluster
import socket
import dask
from collections import defaultdict
# from dask.context import _globals as dask_globals

_globals = defaultdict(lambda: None)
_whitelist = ['client']


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
    c = _globals["client"]
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
        _globals['client'] = c
    return c