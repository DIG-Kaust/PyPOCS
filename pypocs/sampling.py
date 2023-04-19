import numpy as np


def irregular(nr, perc_sub=0.4, seed=10):
    """Random irregular sampling along one direction

    Create indices to perform random irregular sampling
    along one direction

    Parameters
    ----------
    nr : :obj:`int`
        Number of receivers
    perc_sub : :obj:`float`, optional
        Percentage of subsampling
    seed : :obj:`int`, optional
        Random seed

    Returns
    -------
    iava : :obj:`numpy.ndarray`
        Selected indices

    """
    np.random.seed(seed)
    nsub = int(np.round(nr * perc_sub))
    iava = np.sort(np.random.permutation(np.arange(nr))[:nsub])
    return iava


def dithered_irregular(nr, factor_sub=4, seed=10):
    """Dithered irregular sampling along one direction

    Create indices to perform dithered irregular sampling
    along one direction

    Parameters
    ----------
    nr : :obj:`int`
        Number of receivers
    factor_sub : :obj:`int`, optional
        Factor of subsampling
    seed : :obj:`int`, optional
        Random seed

    Returns
    -------
    iava : :obj:`numpy.ndarray`
        Selected indices
    iava_reg : :obj:`numpy.ndarray`
        Indices of regularly subsampled grid (prior to dithering)

    """
    np.random.seed(seed)
    iava_reg = np.arange(nr)[::factor_sub]
    # create dither code
    dither = np.random.randint(-factor_sub // 2 + 1, factor_sub // 2 + 1, len(iava_reg))
    dither[0] = np.random.randint(0, factor_sub // 2 + 1)
    dither[-1] = np.random.randint(-factor_sub // 2 + 1, 0)
    # create locations
    iava = iava_reg + dither
    return iava, iava_reg


def irregular2(nry, nrx, nt, perc_sub=0.4, seed=10):
    """Random irregular sampling along two directions

    Create indices to perform random irregular sampling
    along two directions

    Parameters
    ----------
    nry : :obj:`int`
        Number of receivers along y axis
    nrx : :obj:`int`
        Number of receivers along x axis
    nt : :obj:`int`
        Number of time samples of data to which subsampling will be applied
    perc_sub : :obj:`float`, optional
        Percentage of subsampling
    seed : :obj:`int`, optional
        Random seed

    Returns
    -------
    iava : :obj:`numpy.ndarray`
        Selected indices (including time axis) to be used to decimate the data
    iavarec : :obj:`numpy.ndarray`
        Selected indices along receiver grid

    """
    np.random.seed(seed)
    nsub = int(np.round(nry * nrx * perc_sub))
    iavarec = np.sort(np.random.permutation(np.arange(nry * nrx))[:nsub]).astype(np.int)

    # create mask
    mask = np.zeros((nry * nrx, nt))
    mask[iavarec, :] = 1
    mask = mask.reshape(nry, nrx, nt)
    iava = np.where(mask.ravel() == 1)[0]
    return iava, iavarec


def dithered_irregular2(nry, nrx, nt, factor_sub=3, seed=10):
    """Dithered irregular sampling along two directions

    Create indices to perform dithered irregular sampling
    along two directions

    Parameters
    ----------
    nry : :obj:`int`
        Number of receivers along y axis
    nrx : :obj:`int`
        Number of receivers along x axis
    nt : :obj:`int`
        Number of time samples of data to which subsampling will be applied
    factor_sub : :obj:`int`, optional
        Factor of subsampling
    seed : :obj:`int`, optional
        Random seed

    Returns
    -------
    iava : :obj:`numpy.ndarray`
        Selected indices (including time axis) to be used to decimate the data
    iava2d : :obj:`numpy.ndarray`
        Selected indices along receiver grid
    iava2d_reg : :obj:`numpy.ndarray`
        Indices of regularly subsampled grid (prior to dithering) along receiver grid
    dither : :obj:`numpy.ndarray`
        Dithers

    """
    np.random.seed(seed)
    iava_regy, iava_regx = np.arange(nry)[::factor_sub], np.arange(nrx)[::factor_sub]

    iava_regy, iava_regx = np.meshgrid(iava_regy, iava_regx, indexing='ij')
    iava_reg = np.vstack((iava_regy.ravel(), iava_regx.ravel()))
    nr_reg = iava_reg.shape[1]

    # create dither code
    dithery = np.random.randint(-factor_sub // 2 + 1, factor_sub // 2 + 1, nr_reg)
    ditherx = np.random.randint(-factor_sub // 2 + 1, factor_sub // 2 + 1, nr_reg)
    dither = np.vstack((dithery.ravel(), ditherx.ravel()))

    # create locations
    iavarec = iava_reg + dither

    # create 2d reg mask
    mask = np.zeros((nry, nrx))
    mask[iava_reg[0], iava_reg[1]] = 1
    mask = mask.reshape(nry, nrx)
    iava2d_reg = np.where(mask.ravel() == 1)[0]

    # create 2d mask
    mask = np.zeros((nry, nrx))
    mask[iavarec[0], iavarec[1]] = 1
    mask = mask.reshape(nry, nrx)
    iava2d = np.where(mask.ravel() == 1)[0]

    # create mask
    mask = np.zeros((nry, nrx, nt))
    mask[iavarec[0], iavarec[1], :] = 1
    iava = np.where(mask.ravel() == 1)[0]
    return iava, iava2d, iava2d_reg, dither
