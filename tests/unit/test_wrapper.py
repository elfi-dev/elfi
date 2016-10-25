import numpy as np
from functools import partial

from elfi.wrapper import Wrapper

class Test_wrapper():

    def test_echo_exec_arg(self):
        command = "echo"
        wrapper = Wrapper(command, postprocessor=int)
        ret = wrapper.execute("1")
        assert ret == 1

    def test_echo_default_arg(self):
        command = "echo"
        args = "1"
        wrapper = Wrapper(command, args, postprocessor=int)
        ret = wrapper.execute()
        assert ret == 1

    def test_echo_both_args(self):
        command = "echo"
        args = "1"
        postprocessor = partial(np.fromstring, sep=" ")
        wrapper = Wrapper(command, args, postprocessor=postprocessor)
        ret = wrapper.execute("2")
        assert np.array_equal(ret, np.array([1,2]))

