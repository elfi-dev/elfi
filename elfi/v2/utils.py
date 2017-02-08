import scipy.stats as ss


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