import os
import shutil

import numpy as np


class FileStore:

    def __init__(self, outputs=None, model_name=None, basepath=None):
        self.outputs = None
        self.model_name = model_name
        self.basepath = basepath or os.path.join(os.path.expanduser('~'), '.elfi')

        os.makedirs(self.basepath, exist_ok=True)

    def add_batch(self, batch_output, batch_index):
        outputs = self.outputs or batch_output.keys()
        save_outputs = {k:batch_output[k] for k in outputs}
        filepath = self.filepath(batch_index)
        np.savez(filepath, **save_outputs)

    def read_batch(self, batch_index):
        filepath = self.filepath(batch_index)
        return np.load(filepath)

    @property
    def path(self):
        if not self.model_name:
            raise ValueError('Model name is not specified')
        return os.path.join(self.basepath, self.model_name)

    def filepath(self, batch_index):
        filename = 'batch_' + str(batch_index) + '.npz'
        os.makedirs(self.path, exist_ok=True)
        return os.path.join(self.path, filename)

    def destroy(self):
        shutil.rmtree(self.path)
