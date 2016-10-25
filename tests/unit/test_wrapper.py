import numpy as np
from functools import partial

from elfi.wrapper import Wrapper

class Test_wrapper():

    def test_echo_exec_arg(self):
        command = "echo {0}"
        wrapper = Wrapper(command, post=int)
        ret = wrapper("1")
        assert ret == 1

    def test_echo_default_arg(self):
        command = "echo 1"
        wrapper = Wrapper(command, post=int)
        ret = wrapper()
        assert ret == 1

    def test_echo_both_args(self):
        command = "echo 1 {0}"
        post = partial(np.fromstring, sep=" ")
        wrapper = Wrapper(command, post=post)
        ret = wrapper("2")
        assert np.array_equal(ret, np.array([1,2]))

    def test_echo_kwargs(self):
        command = "echo {param}"
        wrapper = Wrapper(command, post=int)
        ret = wrapper(param="1")
        assert ret == 1

    def test_echo_args_and_kwargs(self):
        command = "echo {param} {0}"
        post = partial(np.fromstring, sep=" ")
        wrapper = Wrapper(command, post=post)
        ret = wrapper(2, param="1")
        assert np.array_equal(ret, np.array([1,2]))

    def test_echo_non_string_args(self):
        command = "echo {0}"
        wrapper = Wrapper(command, post=int)
        ret = wrapper(1)
        assert ret == 1

    def test_echo_parallel_args(self):
        command = "echo {0} {1} {2}"
        post = partial(np.fromstring, sep=" ")
        par = (0, 1)
        wrapper = Wrapper(command, post=post, par=par)
        ret = wrapper(np.atleast_2d([[1],[2]]), np.atleast_2d([[3],[4]]), 5)
        assert np.array_equal(ret, np.atleast_2d([[1, 3, 5], [2, 4, 5]]))

