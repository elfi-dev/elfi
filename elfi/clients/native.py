import logging

from elfi.executor import Executor
import elfi.client

logger = logging.getLogger(__name__)


def set_as_default():
    elfi.client.reset_default()
    elfi.client.set_default_class(Client)


class Client(elfi.client.ClientBase):
    """
    Responsible for sending computational graphs to be executed in an Executor
    """

    def __init__(self):
        self.submit_queue = list()

    def clear_batches(self):
        del self.submit_queue[:]

    def execute(self, loaded_net, override_outputs=None):
        """Execute the computational graph"""
        return Executor.execute(loaded_net)

    def has_batches(self):
        return len(self.submit_queue) > 0

    def num_cores(self):
        return 1

    def num_pending_batches(self, compiled_net=None, context=None):
        n = 0
        for submitted in self.submit_queue:
            if compiled_net and compiled_net != submitted[1]:
                continue
            elif context and context != submitted[2]:
                continue
            n += len(submitted[0])
        return n

    def submit_batches(self, batches, compiled_net, context):
        if not batches:
            return
        self.submit_queue.append((batches, compiled_net, context))

    def wait_next_batch(self):
        batches, compiled_net, context = self.submit_queue.pop(0)
        batch_index = batches.pop(0)

        batch_net = self.load_data(context, compiled_net, batch_index)

        # Insert back to queue if batches left
        if len(batches) > 0:
            submitted = (batches, compiled_net, context)
            self.submit_queue.insert(0, submitted)

        outputs = self.execute(batch_net)
        return outputs, batch_index


set_as_default()
