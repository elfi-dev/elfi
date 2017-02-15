

class ScipyLikeDistribution:
    """Abstract class for an ELFI compatible random distribution.

    Note that the class signature is a subset of that of `scipy.rv_continuous`
    """

    def __init__(self, name=None):
        """

        Parameters
        ----------
        name : name of the distribution
        """
        self.name = name or self.__class__.__name__

    def rvs(self, *params, size=1, random_state):
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

    def pdf(self, x, *params, **kwargs):
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

    def logpdf(self, x, *params, **kwargs):
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