import numpy as np


def linear(threshmax, threshmin, niter):
    """Linear threshold

    Linear decay from Gao et al, 2010

    Parameters
    ----------
    threshmax : :obj:`float`
        Max threshold at first iteration
    threshmin : :obj:`float`
        Min threshold at last iteration
    niter : :obj:`int`, optional
        Number of iterations

    Returns
    -------
    thresh : :obj:`numpy.ndarray`
        Threshold curve of size ``niter``

    """
    thresh = threshmax - (threshmax - threshmin) * np.arange(niter) / (niter-1)
    return thresh


def logarithmic(threshmax, threshmin, niter):
    """Log decay

    Log decay threshold

    Parameters
    ----------
    threshmax : :obj:`float`
        Max threshold at first iteration
    threshmin : :obj:`float`
        Min threshold at last iteration
    niter : :obj:`int`, optional
        Number of iterations

    Returns
    -------
    thresh : :obj:`numpy.ndarray`
        Threshold curve of size ``niter``

    """
    thresh = (np.log(niter) - np.log(np.arange(niter) + 1)) / np.log(niter) * (threshmax - threshmin) + threshmin
    return thresh


def exponential(threshmax, threshmin, niter):
    """Exp decay

    Exp decay threshold

    Parameters
    ----------
    threshmax : :obj:`float`
        Max threshold at first iteration
    threshmin : :obj:`float`
        Min threshold at last iteration
    niter : :obj:`int`, optional
        Number of iterations

    Returns
    -------
    thresh : :obj:`numpy.ndarray`
        Threshold curve of size ``niter``

    """
    thresh = (np.exp(-0.05 * np.arange(niter))) * threshmax + threshmin - np.exp(-0.05*(niter-1)) * threshmax
    return thresh


def exponential1(threshmax, threshmin, niter):
    """Exp decay

    Exp decay threshold from Gao et al, 2010

    Parameters
    ----------
    threshmax : :obj:`float`
        Max threshold at first iteration
    threshmin : :obj:`float`
        Min threshold at last iteration
    niter : :obj:`int`, optional
        Number of iterations

    Returns
    -------
    thresh : :obj:`numpy.ndarray`
        Threshold curve of size ``niter``

    """
    b = - np.log(threshmax / threshmin) / (niter - 1)
    thresh = threshmax * np.exp(b * np.arange(niter))
    return thresh


def power(threshmax, threshmin, niter):
    """Power decay

    Power decay (equivalent to exponential1)

    Parameters
    ----------
    threshmax : :obj:`float`
        Max threshold at first iteration
    threshmin : :obj:`float`
        Min threshold at last iteration
    niter : :obj:`int`, optional
        Number of iterations

    Returns
    -------
    thresh : :obj:`numpy.ndarray`
        Threshold curve of size ``niter``

    """
    thresh = np.power((threshmin / threshmax), np.arange(niter) / (niter - 1)) * threshmax
    return thresh


def invprop(threshmax, threshmin, niter, q=2):
    """Inversely proportional decay

    Inversely proportional decay from Ge Zi-Jian et al, 2015

    Parameters
    ----------
    threshmax : :obj:`float`
        Max threshold at first iteration
    threshmin : :obj:`float`
        Min threshold at last iteration
    niter : :obj:`int`, optional
        Number of iterations
    niter : :obj:`float`, optional
        Exponent

    Returns
    -------
    thresh : :obj:`numpy.ndarray`
        Threshold curve of size ``niter``

    """
    a = niter ** q * (threshmax - threshmin) / (niter ** q - 1)
    b = (niter ** q * threshmin - threshmax) / (niter ** q - 1)
    thresh = a / np.arange(1, niter + 1)**q + b
    return thresh
