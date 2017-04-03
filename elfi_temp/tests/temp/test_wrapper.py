from functools import partial

import numpy as np

from elfi.temp.wrapper import Wrapper


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

    def test_echo_2d_array_args(self):
        command = "echo {0}"
        wrapper = Wrapper(command, post=int)
        ret = wrapper(np.array([[1]]))
        assert ret == 1

