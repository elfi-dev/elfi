import logging

import elfi.client

logger = logging.getLogger(__name__)


def set_as_default():
    elfi.client.reset_default()
    elfi.client.set_default_class(Client)


class Client(elfi.client.ClientBase):
    pass


set_as_default()
