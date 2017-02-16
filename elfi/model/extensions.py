
class ScipyLikeDistribution:
    """Abstract class for an ELFI compatible random distribution. You can implement this
    as having all methods as classmethods or making an instance. Hence the
    signatures include this, instead of self or cls.

    Note that the class signature is a subset of that of `scipy.rv_continuous`
    """

    def __init__(self, name=None):
        """Constuctor (optional, only if instances are meant to be used)

        Parameters
        ----------
        name : name of the distribution
        """
        self._name = name or self.__class__.__name__

    def rvs(this, *params, size=1, random_state):
        """Random variates

        Parameters
        ----------
        param1, param2, ... : array_like
            Parameter(s) of the distribution
        size : int or tuple of ints, optional
        random_state : RandomState

        Returns
        -------
        rvs : ndarray
            Random variates of given size.
        """
        raise NotImplementedError

    def pdf(this, x, *params, **kwargs):
        """Probability density function at x

        Parameters
        ----------
        x : array_like
           points where to evaluate the pdf
        param1, param2, ... : array_like
           parameters of the model

        Returns
        -------
        pdf : ndarray
           Probability density function evaluated at x
        """
        raise NotImplementedError

    def logpdf(this, x, *params, **kwargs):
        """Log of the probability density function at x.

        Parameters
        ----------
        x : array_like
            points where to evaluate the pdf
        param1, param2, ... : array_like
            parameters of the model
        kwargs

        Returns
        -------
        pdf : ndarray
           Log of the probability density function evaluated at x
        """
        raise NotImplementedError

    @property
    def name(this):
        if hasattr(this, '_name'):
            return this._name
        elif isinstance(this, type):
            return this.__name__
        else:
            return this.__class__.__name__
