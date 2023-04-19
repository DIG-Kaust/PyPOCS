import numpy as np


def metrics(data, dataest, itmin=0, verb=False):
    """Metrics

    Compute RMSE and SNR metrics

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        True data of size ``(nx, nt)``
    dataest : :obj:`numpy.ndarray`
        Reconstructed data of size ``(nx, nt)``
    itminnfft : :obj:`int`, optional
        Index of first time sample used to compute metrics (if ``itmin=0``, use entire time axis)
    verb : :obj:`bool`, optional
        Print metrics on screen

    Returns
    -------
    rmse : :obj:`float`
        Root mean square error
    snr : :obj:`float`
        Signal to noise ratio

    """
    rmse = np.linalg.norm(data[:, :, itmin:] - dataest[:, :, itmin:]) / np.linalg.norm(data[:, :, itmin:])
    snr = 20 * np.log10(1 / rmse)
    if verb: print(f'RMSE={rmse}, SNR={snr}')
    return rmse, snr