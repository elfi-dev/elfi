"""This module implements a multiprocessing client using ipyparallel.

http://ipyparallel.readthedocs.io
"""

import itertools
import logging

import ipyparallel as ipp

import elfi.client

logger = logging.getLogger(__name__)


def set_as_default():
    """Set this as the default client."""
    elfi.client.set_client()
    elfi.client.set_default_class(Client)


class Client(elfi.client.ClientBase):
    """A multiprocessing client using ipyparallel.

    http://ipyparallel.readthedocs.io
    """

    def __init__(self, ipp_client=None, **kwargs):
        """Create an ipyparallel client for ELFI.

        Parameters
        ----------
        ipp_client : ipyparallel.Client, optional
            Use this ipyparallel client with ELFI.

        """
        self.ipp_client = ipp_client or ipp.Client(**kwargs)
        self.view = self.ipp_client.load_balanced_view()

        self.tasks = {}
        self._id_counter = itertools.count()

    def apply(self, kallable, *args, **kwargs):
        """Add `kallable(*args, **kwargs)` to the queue of tasks. Returns immediately.

        Parameters
        ----------
        kallable : callable

        Returns
        -------
        id : int
            Number of the queued task.

        """
        id = self._id_counter.__next__()
        async_res = self.view.apply(kallable, *args, **kwargs)
        self.tasks[id] = async_res
        return id

    def apply_sync(self, kallable, *args, **kwargs):
        """Call and returns the result of `kallable(*args, **kwargs)`.

        Parameters
        ----------
        kallable : callable

        """
        return self.view.apply_sync(kallable, *args, **kwargs)

    def get_result(self, task_id):
        """Return the result from task identified by `task_id` when it arrives.

        Parameters
        ----------
        task_id : int
            Id of the task whose result to return.

        """
        async_result = self.tasks.pop(task_id)
        return async_result.get()

    def is_ready(self, task_id):
        """Return whether task with identifier `task_id` is ready.

        Parameters
        ----------
        task_id : int

        """
        return self.tasks[task_id].ready()

    def remove_task(self, task_id):
        """Remove task with identifier `task_id` from pool.

        Parameters
        ----------
        task_id : int

        """
        async_result = self.tasks.pop(task_id)
        if not async_result.ready():
            # Note: Ipyparallel is only able to abort if the job hasn't started.
            return self.ipp_client.abort(async_result, block=False)

    def reset(self):
        """Stop all worker processes immediately and clear pending tasks.

        Note: Ipyparallel is only able to abort if the job hasn't started.
        """
        self.view.abort(block=False)
        self.tasks.clear()

    @property
    def num_cores(self):
        """Return the number of processes."""
        return len(self.view)


# TODO: use import hook instead? https://docs.python.org/3/reference/import.html
set_as_default()
