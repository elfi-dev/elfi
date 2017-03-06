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

        for node in nx_alphabetical_topological_sort(G):
            attr = G.node[node]
            fn = attr['output']
            logger.debug("Executing {}".format(node))
            if callable(fn):
                G.node[node] = cls._run(fn, node, G)
        result = {k:G.node[k]['output'] for k in G.graph['outputs']}
        return result

    @staticmethod
    def _run(fn, node, G):
        args = []
        kwargs = {}

        for parent_name in G.predecessors(node):
            param = G[parent_name][node]['param']
            output = G.node[parent_name]['output']
            if isinstance(param, int):
                args.append((param, output))
            else:
                kwargs[param] = output

        args = [a[1] for a in sorted(args, key=itemgetter(0))]

        output = fn(*args, **kwargs)

        if not isinstance(output, dict):
            output = dict(output=output)
        return output


def nx_alphabetical_topological_sort(G, nbunch=None, reverse=False):
    """Return a list of nodes in topological sort order.

    Modified version of networkx.topological_sort. The difference is that this version
    will always return the same order for the same graph G given that the nodes
    are either strings or numbers. Nodes will be ordered to alphabetical order before
    being added to the search.

    A topological sort is a nonunique permutation of the nodes
    such that an edge from u to v implies that u appears before v in the
    topological sort order.

    Parameters
    ----------
    G : NetworkX digraph
        A directed graph

    nbunch : container of nodes (optional)
        Explore graph in specified order given in nbunch

    reverse : bool, optional
        Return postorder instead of preorder if True.
        Reverse mode is a bit more efficient.

    Raises
    ------
    NetworkXError
        Topological sort is defined for directed graphs only. If the
        graph G is undirected, a NetworkXError is raised.

    NetworkXUnfeasible
        If G is not a directed acyclic graph (DAG) no topological sort
        exists and a NetworkXUnfeasible exception is raised.

    Notes
    -----
    This algorithm is based on a description and proof in
    The Algorithm Design Manual [1]_ .

    The implementation is adapted from networkx.topological_sort.

    References
    ----------
    .. [1] Skiena, S. S. The Algorithm Design Manual  (Springer-Verlag, 1998).
        http://www.amazon.com/exec/obidos/ASIN/0387948600/ref=ase_thealgorithmrepo/
    """
    if not G.is_directed():
        raise nx.NetworkXError(
            "Topological sort not defined on undirected graphs.")

    # nonrecursive version
    seen = set()
    order = []
    explored = set()

    if nbunch is None:
        # Sort them to alphabetical order
        nbunch = sorted(G.nodes())
    for v in nbunch:     # process all vertices in G
        if v in explored:
            continue
        fringe = [v]   # nodes yet to look at
        while fringe:
            w = fringe[-1]  # depth first search
            if w in explored:  # already looked down this branch
                fringe.pop()
                continue
            seen.add(w)     # mark as seen
            # Check successors for cycles and for new nodes
            new_nodes = []
            for n in sorted(G[w]):
                if n not in explored:
                    if n in seen:  # CYCLE !!
                        raise nx.NetworkXUnfeasible("Graph contains a cycle.")
                    new_nodes.append(n)
            if new_nodes:   # Add new_nodes to fringe
                fringe.extend(new_nodes)
            else:           # No new nodes so w is fully explored
                explored.add(w)
                order.append(w)
                fringe.pop()    # done considering this node
    if reverse:
        return order
    else:
        return list(reversed(order))
