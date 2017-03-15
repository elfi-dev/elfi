import logging

import ipyparallel as ipp

from elfi.executor import Executor
import elfi.client

logger = logging.getLogger(__name__)


# TODO: use import hook instead? https://docs.python.org/3/reference/import.html
def set_as_default():
    elfi.client.reset_default()
    elfi.client.set_default_class(Client)


class Client(elfi.client.ClientBase):

    def __init__(self, ipp_client=None):
        self.ipp_client = ipp_client or ipp.Client()
        self.view = self.ipp_client.load_balanced_view()
        self.async_result_list = []

    def clear_batches(self):
        # TODO: currently does not stop jobs being computed in the cluster
        self.view.abort()
        del self.async_result_list[:]

    def execute(self, loaded_net):
        return self.view.apply_sync(Executor.execute, loaded_net)

    def has_batches(self):
        return self.num_pending_batches() > 0

    def num_pending_batches(self, compiled_net=None, context=None):
        return len(self.async_result_list)

    def num_cores(self):
        return len(self.view)

    def submit_batches(self, batches, compiled_net, context):
        if not batches:
            return

        for batch_index in batches:
            batch_net = self.load_data(context, compiled_net, batch_index)
            async_res = self.view.apply(Executor.execute, batch_net)
            self.async_result_list.append((async_res, batch_index))

    def wait_next_batch(self, async=False):
        # TODO: async operation, check ipyparallel.client.asyncresult _unordered_iter
        tup = self.async_result_list.pop(0)
        return tup[0].get(), tup[1]


# TODO: use import hook instead? https://docs.python.org/3/reference/import.html
set_as_default()
