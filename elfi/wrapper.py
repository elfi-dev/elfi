import sys
import numpy as np
from subprocess import check_output

class Wrapper():
    """ Wraps an external command to work as a callable operation for a node

        Should work with interfaces:
        - data = simulator(N, *input_dict['data'], prng=prng)
        - data = operation(*input['data'])
        - data = operation(input['data'], input['observed'])

        Parameters
        ----------
        command: string
            executable command
        args: list of tuples
            command line arguments for command, in order
            tuples should be in order, and contain values
              - unique name
              - 1 if kwarg, else 0
              - default value, or None
        postprocessor: function(string)
            function that processes the stdout (as string)
    """

    def __init__(self, command_template="", post=None, pre=None, par=None):
        if len(command_template) < 1:
            raise ValueError("Not a valid command")
        self.command_template = command_template
        self.post = post or self.read_nparray
        self.pre = pre or self.process_seed
        self.par = par or list()

    @staticmethod
    def process_seed(command_template, args, kwargs):
        """ Replace 'prng' in kwargs with a seed from the generator if present in template """
        if "prng" in kwargs.keys():
            if "{_seed" in command_template:
                if isinstance(kwargs["prng"], np.random.RandomState):
                    kwargs["_seed"] = str(kwargs["prng"].randint(np.iinfo(np.int32).max))
            del kwargs["prng"]
        return command_template, args, kwargs

    @staticmethod
    def read_nparray(stdout):
        """ Interpret the stdout as a space-separated numpy array """
        return np.fromstring(stdout, sep=" ")

    def __call__(self, *args, **kwargs):
        """ Executes the wrapped command, with additional arguments and keyword arguments.

        Returns
        -------
        if regular input data:
            postprocessed stdout from executing command
        if parallel input data:
            numpy 2d array with parallel simulation results in rows
        """
        if len(self.par) < 1:
            # no parallel arguments
            return self._run(*args, **kwargs)
        else:
            # execute parallel arguments sequentially
            ret = None
            npar = len(args[self.par[0]])
            for i in range(npar):
                argsi = list()
                for j, arg in enumerate(args):
                    if j in self.par:
                        # assume 2d array
                        argsi.append(arg[i][0])
                    else:
                        argsi.append(arg)
                if ret is None:
                    ret = self._run(*argsi, **kwargs)
                else:
                    ret = np.vstack((ret, self._run(*argsi, **kwargs)))
            return ret


    def _run(self, *args, **kwargs):
        template, args, kwargs = self.pre(self.command_template, args, kwargs)
        command = template.format(*args, **kwargs)
        argv = command.split(" ")
        stdout = check_output(argv)
        ret = self.post(stdout)
        return ret

