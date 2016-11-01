import numpy as np
from elfi.bo.gpy_model import GpyModel

class Test_GpyModel():

    def test_default_init(self):
        gp = GpyModel()
        assert gp.n_observations() == 0
        assert gp.evaluate(np.zeros(1)) == (0.0, 0.0, 0.0)
        assert gp.evaluate(np.ones(1)) == (0.0, 0.0, 0.0)


