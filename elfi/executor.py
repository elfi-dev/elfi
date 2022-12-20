"""This module includes the Executor of ELFI graphs."""

import logging
from operator import itemgetter

import networkx as nx

logger = logging.getLogger(__name__)


class Executor:
    """Responsible for computing the graph G.

    The format of the computable graph G is `nx.DiGraph`. The execution order of the nodes
    is fixed and follows the topological ordering of G. The following properties are
    required.

    ### Keys in G.graph dictionary

    outputs : list
        lists all the names of the nodes whose outputs are returned.


    ### Keys in edge dictionaries, G[parent_name][child_name]

    param : str or int
        The parent node output is passed as a parameter with this name to the child node.
        Integers are interpreted as positional parameters.


    ### Keys in node dictionaries, G.node

    operation : callable
        Executed with with the parameter specified in the incoming edges
    output : variable
        Existing output value taken as an output itself

    Notes
    -----
    You cannot have both operation and output in the same node dictionary

    """

    @classmethod
    def execute(cls, G):
        """Execute a graph.

        Parameters
        ----------
        G : nx.DiGraph

        Returns
        -------
        dict of node outputs

        """
        order = cls.get_execution_order(G)

        for node in order:
            attr = G.nodes[node]
            logger.debug("Executing {}".format(node))

            if attr.keys() >= {'operation', 'output'}:
                raise ValueError('Generative graph has both op and output present for '
                                 'node {}'.format(node))

            if 'operation' in attr:
                op = attr['operation']
                try:
                    G.nodes[node].update(cls._run(op, node, G))
                    del G.nodes[node]['operation']
                except Exception as exc:
                    raise exc.__class__("In executing node '{}': {}."
                                        .format(node, exc)).with_traceback(exc.__traceback__)

            elif 'output' not in attr:
                raise ValueError('Generative graph has no op or output present for node '
                                 '{}'.format(node))

        # Make a result dict based on the requested outputs
        result = {k: G.nodes[k]['output'] for k in G.graph['outputs']}
        return result

    @classmethod
    def get_execution_order(cls, G):
        """Return a list of nodes to execute.

        This method returns the minimal list of nodes that need to be executed in
        graph G in order to return the requested outputs.

        The ordering of the nodes is fixed.

        Parameters
        ----------
        G : nx.DiGraph

        Returns
        -------
        nodes : list
            nodes that require execution

        """
        # Get the cache dict if it exists
        cache = G.graph.get('_executor_cache', {})

        output_nodes = G.graph['outputs']
        # Filter those output nodes who have an operation to run
        needed = tuple(sorted(node for node in output_nodes if 'operation' in G.nodes[node]))

        if len(needed) == 0:
            return []

        if needed not in cache:
            # Resolve the nodes that need to be executed in the graph
            nodes_to_execute = set(needed)

            if 'sort_order' not in cache:
                cache['sort_order'] = nx_constant_topological_sort(G)
            sort_order = cache['sort_order']

            # Resolve the dependencies of needed
            dep_graph = nx.DiGraph(G.edges)
            for node in sort_order:
                attr = G.nodes[node]
                if attr.keys() >= {'operation', 'output'}:
                    raise ValueError('Generative graph has both op and output present')

                # Remove those nodes from the dependency graph whose outputs are present
                if 'output' in attr:
                    dep_graph.remove_node(node)
                elif 'operation' not in attr:
                    raise ValueError('Generative graph has no op or output present')

            # Add the dependencies of the needed nodes
            for needed_node in needed:
                nodes_to_execute.update(nx.ancestors(dep_graph, needed_node))

            # Turn in to a sorted list and cache
            cache[needed] = [n for n in sort_order if n in nodes_to_execute]

        return cache[needed]

    @staticmethod
    def _run(fn, node, G):
        args = []
        kwargs = {}

        for parent_name in G.predecessors(node):
            param = G[parent_name][node]['param']
            output = G.nodes[parent_name]['output']
            if isinstance(param, int):
                args.append((param, output))
            else:
                kwargs[param] = output

        args = [a[1] for a in sorted(args, key=itemgetter(0))]

        output_dict = {'output': fn(*args, **kwargs)}
        return output_dict


def nx_constant_topological_sort(G, nbunch=None, reverse=False):
    """Return a list of nodes in a constant topological sort order.

    This implementations is adapted from `networkx.topological_sort`.

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
    This algorithm is based on a description and proof in The Algorithm Design
    Manual [1].

    References
    ----------
    .. [1] Skiena, S. S. The Algorithm Design Manual  (Springer-Verlag, 1998).
        http://www.amazon.com/exec/obidos/ASIN/0387948600/ref=ase_thealgorithmrepo/

    """
    if not G.is_directed():
        raise nx.NetworkXError("Topological sort not defined on undirected graphs.")

    # nonrecursive version
    seen = set()
    order = []
    explored = set()

    if nbunch is None:
        # Sort them to alphabetical order
        nbunch = sorted(G.nodes())
    for v in nbunch:  # process all vertices in G
        if v in explored:
            continue
        fringe = [v]  # nodes yet to look at
        while fringe:
            w = fringe[-1]  # depth first search
            if w in explored:  # already looked down this branch
                fringe.pop()
                continue
            seen.add(w)  # mark as seen
            # Check successors for cycles and for new nodes
            new_nodes = []
            for n in sorted(G[w]):
                if n not in explored:
                    if n in seen:  # CYCLE !!
                        raise nx.NetworkXUnfeasible("Graph contains a cycle.")
                    new_nodes.append(n)
            if new_nodes:  # Add new_nodes to fringe
                fringe.extend(new_nodes)
            else:  # No new nodes so w is fully explored
                explored.add(w)
                order.append(w)
                fringe.pop()  # done considering this node
    if reverse:
        return order
    else:
        return list(reversed(order))
