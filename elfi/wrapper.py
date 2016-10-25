import sys
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
        args: list of strings
            positional arguments for command, in order
        kwargs: dictionary of (keyword: argument) pairs
            keyword arguments for command
        postprocessor: function(string)
            function that processes the stdout (as string)
    """

    def __init__(self, command, args=list(), kwargs=dict(), postprocessor=None):
        self.command = command
        self.args = args
        self.kwargs = kwargs
        self.postprocessor = postprocessor

    @staticmethod
    def _add_args(argv, *args):
        argv.extend(args)
        return argv

    @staticmethod
    def _add_kwargs(argv, **kwargs):
        for k, v in kwargs:
            if len(k) > 1:
                argv.append("--%s" % (k))
            else:
                argv.append("-%s" % (k))
            argv.append(v)
        return argv

    def execute(self, *args, **kwargs):
        """ Executes the wrapped command, with additional arguments and keyword arguments.
        """
        argv = [self.command,]
        argv = Wrapper._add_args(argv, *self.args)
        argv = Wrapper._add_args(argv, *args)
        argv = Wrapper._add_kwargs(argv, **self.kwargs)
        argv = Wrapper._add_kwargs(argv, **kwargs)
        print("Executing: %s" % (argv), file=sys.stderr)
        ret = check_output(argv)
        if self.postprocessor is not None:
            ret = self.postprocessor(ret)
            print("Result (processed): %s" % (ret), file=sys.stderr)
        else:
            print("Result: %s" % (ret), file=sys.stderr)
        return ret

