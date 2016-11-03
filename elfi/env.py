from distributed import Client
import dask

__all__ = ['env']

class Environment():
    def __init__(self):
        self.client = Client()
        dask.set_options(get=self.client.get)

env = Environment()