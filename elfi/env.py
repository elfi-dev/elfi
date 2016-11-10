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
    return _globals[key]


def client():
    if _globals['client'] is None:
        # Do not start the diagnostics (bokeh) by default
        try:
            cluster = LocalCluster(diagnostics_port=None)
        except (OSError, socket.error):
            cluster = LocalCluster(scheduler_port=0, diagnostics_port=None)
        _globals['client'] = Client(cluster, set_as_default=True)
    return _globals['client']