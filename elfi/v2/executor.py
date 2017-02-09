import logging
from operator import itemgetter

import networkx as nx
import numpy as np


logger = logging.getLogger(__name__)


class Client:
    """
    Responsible for sending computational graphs to be executed in an Executor
    """

    def execute(self, G):
        """Execute the computational graph"""
        pass


class Executor:
    """
    Responsible for computing the graph
    """

    @classmethod
    def execute(cls, G):
        for nodename in nx.topological_sort(G):
            attr = G.node[nodename]
            fn = attr['output']
            logger.debug("Executing {}".format(nodename))
            if callable(fn):
                G.node[nodename] = cls._run(fn, nodename, G)
        return G

    @staticmethod
    def _run(fn, nodename, G):
        args = []
        kwargs = {}

        for parent_name in G.predecessors(nodename):
            param = G[parent_name][nodename]['param']
            output = G.node[parent_name]['output']
            if isinstance(param, int):
                args.append((param, output))
            else:
                kwargs[param] = output

        args = [a[1] for a in sorted(args, key=itemgetter(0))]
        kwargs['n'] = G.graph['n']

        output = fn(*args, **kwargs)
        if not isinstance(output, dict):
            output = dict(output=output)
        return output


"""TODO: Below to be removed"""


class LegacyExecutor:
    """
    Responsible for computing the graph
    """

    def execute(self, G):
        for nodename in nx.topological_sort(G):
            node = G.node[nodename]
            print("Executing {}".format(nodename))
            if callable(node['output']):
                node['output'] = self._evaluate_node(node, nodename, G)

    # TODO: replace so that transforms receive the outputs from their respective nodes directly
    #       without intermediate input_dict phase
    def _evaluate_node(self, node, nodename, G):
        data = []; obs = []
        for parent in G.predecessors(nodename):
            parent_output = G.node[parent]['output']
            data.append(parent_output['data'])
            obs.append(G.node[parent]['observed'])
        data = tuple(data); obs = tuple(obs)

        span = G.graph['batch_span']
        # TODO: remove input_dict phase
        input_dict = {
            "data": data,
            "n": span[1] - span[0],
            "index": span[0],
            "random_state": np.random.RandomState(0).get_state(),
            "observed": obs
        }
        return node['output'](input_dict)


# Some sketching for nodes

"""

transform(dependency_outputs, instructions=instructions)


Instructions:

operation
observed_data
initial_random_state

State:

span
current_index
current_random_state


Environment:

require: 'matlab'

"""



"""

[transform(1, ...), transform(2, ...]


{


}


"""





























