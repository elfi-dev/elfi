"""This module implements a multiprocessing client using dask."""

import itertools
import os

from dask.distributed import Client as DaskClient

import elfi.client


def set_as_default():
    """Set this as the default client."""
    elfi.client.set_client()
    elfi.client.set_default_class(Client)


class Client(elfi.client.ClientBase):
    """A multiprocessing client using dask."""

    def __init__(self):
        """Initialize a dask client."""
        self.dask_client = DaskClient()
        self.tasks = {}
        self._id_counter = itertools.count()

    def apply(self, kallable, *args, **kwargs):
        """Add `kallable(*args, **kwargs)` to the queue of tasks. Returns immediately.

        Parameters
        ----------
        kallable: callable

        Returns
        -------
        task_id: int

        """
        task_id = self._id_counter.__next__()
        async_result = self.dask_client.submit(kallable, *args, **kwargs)
        self.tasks[task_id] = async_result
        return task_id

    def apply_sync(self, kallable, *args, **kwargs):
        """Call and returns the result of `kallable(*args, **kwargs)`.

        Parameters
        ----------
        kallable: callable

        """
        return self.dask_client.run_on_scheduler(kallable, *args, **kwargs)

    def get_result(self, task_id):
        """Return the result from task identified by `task_id` when it arrives.

        Parameters
        ----------
        task_id: int

        Returns
        -------
        dict

        """
        async_result = self.tasks.pop(task_id)
        return async_result.result()

    def is_ready(self, task_id):
        """Return whether task with identifier `task_id` is ready.

        Parameters
        ----------
        task_id: int

        Returns
        -------
        bool

        """
        return self.tasks[task_id].done()

    def remove_task(self, task_id):
        """Remove task with identifier `task_id` from scheduler.

        Parameters
        ----------
        task_id: int

        """
        async_result = self.tasks.pop(task_id)
        if not async_result.done():
            async_result.cancel()

    def reset(self):
        """Stop all worker processes immediately and clear pending tasks."""
        self.dask_client.shutdown()
        self.tasks.clear()

    @property
    def num_cores(self):
        """Return the number of processes.

        Returns
        -------
        int

        """
        return os.cpu_count()


set_as_default()
