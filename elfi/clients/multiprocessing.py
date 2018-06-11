"""This module implements a simple multiprocessing client."""

import itertools
import logging
import multiprocessing

import elfi.client

logger = logging.getLogger(__name__)


def set_as_default():
    """Set this as the default client."""
    elfi.client.set_client()
    elfi.client.set_default_class(Client)


class Client(elfi.client.ClientBase):
    """Client based on Python's built-in multiprocessing module."""

    def __init__(self, num_processes=None, **kwargs):
        """Create a multiprocessing client.

        Parameters
        ----------
        num_processes : int, optional
            Number of worker processes to use. Defaults to os.cpu_count().

        """
        num_processes = num_processes or kwargs.pop('processes', None)
        self.pool = multiprocessing.Pool(processes=num_processes, **kwargs)

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
        async_res = self.pool.apply_async(kallable, args, kwargs)
        self.tasks[id] = async_res
        return id

    def apply_sync(self, kallable, *args, **kwargs):
        """Call and returns the result of `kallable(*args, **kwargs)`.

        Parameters
        ----------
        kallable : callable

        """
        return self.pool.apply(kallable, args, kwargs)

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
        if task_id in self.tasks:
            del self.tasks[task_id]
            # TODO: also kill the pid?

    def reset(self):
        """Stop all worker processes immediately and clear pending tasks."""
        self.pool.terminate()
        self.pool.join()
        self.tasks.clear()

    @property
    def num_cores(self):
        """Return the number of processes."""
        return self.pool._processes  # N.B. Not necessarily the number of actual cores.


# TODO: use import hook instead? https://docs.python.org/3/reference/import.html
set_as_default()
