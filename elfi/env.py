from collections import defaultdict
import socket
import logging

from distributed import Client, LocalCluster
from elfi.inference_task import InferenceTask

_globals = defaultdict(lambda: None)
_whitelist = ["client", "inference_task"]

logging.getLogger('tornado').setLevel(logging.WARNING)


def set_option(**kwargs):
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
            raise ValueError("Unrecognized ELFI environment setting {}".format(k))
        _globals[k] = kwargs[k]


def get_option(key):
    if key not in _whitelist:
        raise ValueError("Unrecognized ELFI environment setting {}".format(key))
    return _globals[key]


def clear_option(key):
    if key not in _whitelist:
        raise ValueError("Unrecognized ELFI environment setting {}".format(key))
    if key in _globals:
        del _globals[key]


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

    c = get_option("client")
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
        set_option(client = c)
    return c


def inference_task(default=None):
    """Returns the current `InferenceTask`. If default class is provided
    creates a new instance with a given name if none is found.

    Parameters
    ----------
    default : InferenceTask (optional)
       sets default as the current inference task

    Returns
    -------
    `InferenceTask`

    """
    if default is not None:
        if not isinstance(default, InferenceTask):
            raise ValueError("Parameter default must be of type InferenceTask")
        set_option(inference_task=default)

    itask = get_option("inference_task")
    if itask is None:
        # Create a new default inference task
        itask = InferenceTask()
        set_option(inference_task=itask)
    return itask


def new_inference_task(*args, **kwargs):
    """Sets a new inference task for the environment.

    Useful when you want to start anew without worrying about previous node names or
    results.

    Examples
    --------
    p = Prior('p', 'uniform')
    # `p` will now be left to the earlier inference task
    env.new_inference_task()
    # Now we can create a new node with the same name, and it will go to our new
    # inference task
    p = Prior('p', 'uniform')

    Parameters
    ----------
    The parameters will be passed to `InferenceTask`

    Returns
    -------
    `InferenceTask`
    """

    itask = InferenceTask(*args, **kwargs)
    return inference_task(itask)
