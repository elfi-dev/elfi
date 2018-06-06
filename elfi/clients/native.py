"""This module implements the native single-core client."""

import itertools
import logging

import elfi.client

logger = logging.getLogger(__name__)


def set_as_default():
    """Set this as the default client."""
    elfi.client.set_client()
    elfi.client.set_default_class(Client)


class Client(elfi.client.ClientBase):
    """Simple non-parallel client.

    Responsible for sending computational graphs to be executed in an Executor
    """

    def __init__(self, **kwargs):
        """Create a native client."""
        self.tasks = {}
        self._ids = itertools.count()

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
        id = self._ids.__next__()
        self.tasks[id] = (kallable, args, kwargs)
        return id

    def apply_sync(self, kallable, *args, **kwargs):
        """Call and returns the result of `kallable(*args, **kwargs)`.

        Parameters
        ----------
        kallable : callable

        """
        return kallable(*args, **kwargs)

    def get_result(self, task_id):
        """Return the result from task identified by `task_id` when it arrives.

        Parameters
        ----------
        task_id : int
            Id of the task whose result to return.

        """
        kallable, args, kwargs = self.tasks.pop(task_id)
        return kallable(*args, **kwargs)

    def is_ready(self, task_id):
        """Return whether task with identifier `task_id` is ready.

        Parameters
        ----------
        task_id : int

        """
        return True

    def remove_task(self, task_id):
        """Remove task with identifier `task_id` from pool.

        Parameters
        ----------
        task_id : int

        """
        if task_id in self.tasks:
            del self.tasks[task_id]

    def reset(self):
        """Stop all worker processes immediately and clear pending tasks."""
        self.tasks.clear()

    @property
    def num_cores(self):
        """Return the number of processes, which is always 1 for the native client."""
        return 1


set_as_default()
