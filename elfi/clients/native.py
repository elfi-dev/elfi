import logging
import itertools


from elfi.executor import Executor
import elfi.client

logger = logging.getLogger(__name__)


def set_as_default():
    elfi.client.set_client()
    elfi.client.set_default_class(Client)


class Client(elfi.client.ClientBase):
    """
    Responsible for sending computational graphs to be executed in an Executor
    """

    def __init__(self):
        self.tasks = {}
        self._ids = itertools.count()

    def apply(self, kallable, *args, **kwargs):
        id = self._ids.__next__()
        self.tasks[id] = (kallable, args, kwargs)
        return id

    def apply_sync(self, kallable, *args, **kwargs):
        return kallable(*args, **kwargs)

    def get_result(self, task_id):
        kallable, args, kwargs = self.tasks.pop(task_id)
        return kallable(*args, **kwargs)

    def is_ready(self, task_id):
        return True

    def remove_task(self, task_id):
        if task_id in self.tasks:
            del self.tasks[task_id]

    def reset(self):
        self.tasks.clear()

    @property
    def num_cores(self):
        return 1

set_as_default()
