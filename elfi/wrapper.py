import sys
import numpy as np
from subprocess import check_output

class Wrapper():
    """ Wraps an external command to work as a callable operation for a node.

        Currently only supports sequential operations (not vectorized).
        You can force this by setting batch_size=1 in the ABC method.

        Parameters
        ----------
        command_template: string
            python format template for command
        post: function(string)
            function that processes the stdout (as string)
        pre: function(command_template, args, kwargs)
            function that preprocesses the template, args and kwargs
    """

    def __init__(self, command_template="", post=None, pre=None):
        if len(command_template) < 1:
            raise ValueError("Not a valid command")
        self.command_template = command_template
        self.post = post or self.read_nparray
        self.pre = pre or self.process_elfi_internals

    @staticmethod
    def process_elfi_internals(command_template, args, kwargs):
        """ Replace 'random_state' in kwargs with a seed from the generator if present in template """
        proc_args = list()
        for a in args:
            if isinstance(a, np.ndarray):
                if a.shape[0] == 1:
                    # take single values out of array
                    proc_args.append(a.item())
                else:
                    raise NotImplementedError("Wrapper does not yet support array arguments")
            else:
                proc_args.append(a)
        if "random_state" in kwargs.keys():
            if "{seed}" in command_template:
                if isinstance(kwargs["random_state"], np.random.RandomState):
                    kwargs["seed"] = str(kwargs["random_state"].randint(np.iinfo(np.uint32).max))
            del kwargs["random_state"]
        return command_template, proc_args, kwargs

    @staticmethod
    def read_nparray(stdout):
        """ Interpret the stdout as a space-separated numpy array """
        return np.fromstring(stdout, sep=" ")[None, :]

    def __call__(self, *args, **kwargs):
        """ Executes the wrapped command, with additional arguments and keyword arguments.

        Arguments
        ---------
        formatting arguments for command template

        Returns
        -------
        postprocessed stdout from executing command
        """
        template, args, kwargs = self.pre(self.command_template, args, kwargs)
        command = template.format(*args, **kwargs)
        argv = command.split(" ")
        stdout = check_output(argv, universal_newlines=True)
        return self.post(stdout)

