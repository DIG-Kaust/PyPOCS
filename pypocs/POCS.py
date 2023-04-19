import numpy as np
from pylops.basicoperators import Diagonal, Identity


def POCS(data, mask, SOp, opdims, thresh, niter,
         BOp=None, freqwavscaling=None, SOp1=None, dinit=None,
         dtrue=None, masktrue=None, show=False, history=True):
    """POCS interpolation

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        Data of size ``(nx, nt)`` or ``(ny, nx, nt)``
    mask : :obj:`numpy.ndarray`
        Mask of size  ``ny`` (subsampling along one axis) or ``(ny, nx, nt)`` (subsampling along two axes).
        When using BOp simply provide a mask of ones with the size of the reconstructed data
    SOp : :obj:`pylops.LinearOperator`
        Sparsifying operator
    opdims : :obj:`tuple`
        Deprecated
    thresh : :obj:`float` or :obj:`numpy.ndarray`
        Constant threshold or iteration dependant threshold of size ``niter``
    niter : :obj:`int`
        Number of iterations
    BOp : :obj:`pylops.LinearOperator`, optional
        Bilinear interpolation to use instead of the restriction operator for off-the-grid receivers.
        If passed, mask will be only used to infer the dimensions of the reconstructed model
    freqwavscaling : :obj:`tuple`, optional
        Parameters to use when downscaling high frequiencies at early iterations.
        Must be provided as ((fmin, fmax), itermax) as it will go from fmin to fmax
        in a linear fashion from iteration 0 to itermax
    SOp1 : :obj:`pylops.LinearOperator`, optional
        Sparsifying operator to additionally add after downweighting high-frequencies
        (when ``freqwavscaling`` is not None)
    dinit : :obj:`numpy.ndarray`, optional
        Data to use as starting guess of size ``(nx, nt)`` or ``(ny, nx, nt)``. If None, ``data`` will be used
        as starting guess
    dtrue : :obj:`numpy.ndarray`, optional
        True data, only used to compute RMSE over iterations
    masktrue : :obj:`numpy.ndarray`, optional
        Mask to apply to updates, only used to compute RMSE over iterations
    show : :obj:`bool`, optional
        Print iterations
    history : :obj:`int`, optional
        Number of steps after which the current solution is collected (use ``-1`` to avoid collecting it)

    """
    # History (for backward compatibility with bool)
    history = -1 if isinstance(history, bool) else history

    if BOp is None:
        Rop = Diagonal(mask.flatten())
        I_Rop = Diagonal(np.abs(1-mask.flatten()))
    else:
        Rop = BOp
        I_Rop = Identity(BOp.shape[1]) - BOp.H * BOp

    # if threshold is not already provided as vector make constant
    if np.array(thresh).size == 1:
        thresh = np.ones(niter) * thresh

    # freqwavnumber scaling
    if freqwavscaling is not None:
        (fmaxin, fmaxend), fwniter = freqwavscaling
        fmax = np.linspace(fmaxin, fmaxend, fwniter)

    # masked data
    datamasked = Rop.H * data.ravel()

    # initial guess
    datainv = data.copy() if dinit is None else dinit.copy()

    # run iterations
    datainv_hist = []
    Dthresh_hist = []
    derr_hist = []
    for i in range(niter):
        if show: print(f'Iteration {i}/{niter}')
        # transform
        Dthresh = SOp.H * datainv.ravel()
        # downweight high-freqs (optional)
        if freqwavscaling is not None and i < fwniter:
            fwmask = np.zeros(SOp.dims)
            fwmask[:, :int(fmax[i])] = 1.
            Dthresh = Dthresh * fwmask.flatten()
        if SOp1 is not None:
            Dthresh = SOp1.H * Dthresh

        # thresh
        Dthresh[np.abs(Dthresh) < thresh[i]] = 0

        # transform back
        if SOp1 is not None:
            Dthresh = SOp1 * Dthresh
        datainv = SOp * Dthresh
        
        # place reconstructed traces where missing in data and true traces where available in data
        datainv = datamasked + I_Rop * datainv.ravel()
        datainv = np.real(datainv.reshape(mask.shape))
        
        # tracking history
        if history != -1 and (i == 0 or (i + 1) % history == 0):
            Dthresh_hist.append(Dthresh)
            datainv_hist.append(datainv)
        if dtrue is not None:
            if masktrue is None:
                derr_hist.append(float(np.linalg.norm(dtrue-datainv) / np.linalg.norm(dtrue)))
            else:
                derr_hist.append(float(np.linalg.norm((dtrue - datainv) * masktrue) / np.linalg.norm(dtrue * masktrue)))

    return datainv, datainv_hist, Dthresh_hist, derr_hist, thresh