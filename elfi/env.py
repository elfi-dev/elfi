from distributed import Client
import dask
from collections import defaultdict
#from dask.context import _globals as dask_globals

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
        _globals['client'] = Client()
        dask.set_options(get=_globals['client'].get)
    return _globals['client']