import logging
import itertools

import ipyparallel as ipp

from elfi.executor import Executor
import elfi.client

logger = logging.getLogger(__name__)


# TODO: use import hook instead? https://docs.python.org/3/reference/import.html
def set_as_default():
    elfi.client.set_client()
    elfi.client.set_default_class(Client)


class Client(elfi.client.ClientBase):

    def __init__(self, ipp_client=None):
        self.ipp_client = ipp_client or ipp.Client()
        self.view = self.ipp_client.load_balanced_view()

        self.tasks = {}
        self._id_counter = itertools.count()

    def apply(self, kallable, *args, **kwargs):
        id = self._id_counter.__next__()
        async_res = self.view.apply(kallable, *args, **kwargs)
        self.tasks[id] = async_res
        return id

    def apply_sync(self, kallable, *args, **kwargs):
        return self.view.apply_sync(kallable, *args, **kwargs)

    def get(self, task_id):
        async_result = self.tasks.pop(task_id)
        return async_result.get()

    def wait_next(self, task_ids):
        # TODO: async operation, check ipyparallel.client.asyncresult _unordered_iter
        raise NotImplementedError

    def is_ready(self, task_id):
        return self.tasks[task_id].ready()

    def remove_task(self, task_id):
        async_result = self.tasks.pop(task_id)
        if not async_result.ready():
            # TODO: Find a way to safely cancel.
            # This freezes the execution if the result arrives about the same time
            # I call this.
            # async_result.abort()
            pass

    def reset(self):
        # Note: does not stop jobs being computed in the cluster as this is not supported in ipyparallel
        self.view.abort()
        self.tasks.clear()

    @property
    def num_cores(self):
        return len(self.view)

# TODO: use import hook instead? https://docs.python.org/3/reference/import.html
set_as_default()
