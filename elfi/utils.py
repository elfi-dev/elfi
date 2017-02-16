import scipy.stats as ss
import networkx as nx

SCIPY_ALIASES = {
    'normal': 'norm',
    'exponential': 'expon',
    'unif': 'uniform',
    'bin': 'binom',
    'binomial': 'binom'
}


def scipy_from_str(name):
    name = name.lower()
    name = SCIPY_ALIASES.get(name, name)
    return getattr(ss, name)


def splen(span):
    return span[1] - span[0]


def args_to_tuple(*args):
    return tuple(args)


# NetworkX utils


def all_ancestors(G, nbunch):
    # Resolve output ancestors
    ancestors = set(nbunch)
    for node in nbunch:
        ancestors = ancestors.union(nx.ancestors(G, node))
    return ancestors


def nx_search_iter(net, start_node, breadth_first=True):
    i_pop = 0 if breadth_first is True else -1
    visited = []
    search = sorted(net.predecessors(start_node))
    while len(search) > 0:
        s = search.pop(i_pop)
        if s in visited:
            continue

        yield s

        visited.append(s)
        found = sorted(net.predecessors(s))
        search += found
