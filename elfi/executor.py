import logging
from operator import itemgetter

import networkx as nx

logger = logging.getLogger(__name__)


class Executor:
    """
    Responsible for computing the graph
    """

    @classmethod
    def execute(cls, G):
        """

        Parameters
        ----------
        G : nx.DiGraph

        Returns
        -------
        dict of node outputs

        """
        for nodename in nx.topological_sort(G):
            attr = G.node[nodename]
            fn = attr['output']
            logger.debug("Executing {}".format(nodename))
            if callable(fn):
                G.node[nodename] = cls._run(fn, nodename, G)
        result = {k:G.node[k] for k in G.graph['outputs']}
        return result

    @staticmethod
    def _run(fn, nodename, G):
        node_attr = G.node[nodename]
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

        # Add requested runtime variables to node
        runtime = node_attr.get('runtime', tuple())
        for key in runtime:
            kwargs[key] = G.graph[key]

        output = fn(*args, **kwargs)

        if not isinstance(output, dict):
            output = dict(output=output)
        return output
