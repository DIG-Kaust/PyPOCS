import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pylops
try:
    import cupy as cp
except:
    print('Cupy not available...')

from pylops.basicoperators import Restriction, Identity
from pylops.signalprocessing import FFTND
from pylops.utils.backend import get_module

from pyproximal.proximal import *
from pyproximal.optimization.primal import *
from pyproximal.optimization.primaldual import *
from pypocs.POCS import POCS
from pypocs import threshold


class Callback():
    """Callback

    Custom callback for solvers to collect intermediate solutions and track error norm

    Parameters
    ----------
    xtrue : :obj:`numpy.ndarray`
        True solution
    history : :obj:`int`, optional
        Number of steps after which the current solution is collected (use ``-1`` to avoid collecting it)
    masktrue : :obj:`numpy.ndarray`, optional
        Mask to apply to the true and estimated solutions (if ``None``, no mask is applied)
    backend : :obj:`str`, optional
        Backend (``numpy`` or ``cupy``)

    """
    def __init__(self, xtrue, history=-1, masktrue=None, backend="numpy"):
        self.xtrue = xtrue
        self.history = history
        self.masktrue = masktrue
        self.backend = backend
        self.err = []
        self.xhist = []
        self.iiter = 0

    def __call__(self, x):
        if self.masktrue is None:
            self.err.append(float(np.linalg.norm(x-self.xtrue) / np.linalg.norm(self.xtrue)))
        else:
            self.err.append(float(np.linalg.norm((x-self.xtrue) * self.masktrue) /
                                  np.linalg.norm(self.xtrue * self.masktrue)))
        if self.history != -1 and (self.iiter == 0 or (self.iiter + 1) % self.history == 0):
            if self.masktrue is not None:
                x = self.masktrue * x
            if self.backend == "numpy":
                self.xhist.append(x)
            else:
                self.xhist.append(cp.asnumpy(x))
        self.iiter += 1


def pocs_interpolate(x, mask, samplings, nfft,
                     thresh, threshkind='linear',
                     BOp=None, MOp=None,
                     niter=10, xinit=None, xtrue=None, masktrue=None,
                     history=True, backend="numpy", verb=False):
    """POCS interpolator

    Wrapper function to run POCS interpolation in :func:`pypocs.pocs.POCS`

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Data of size ``(nx, nt)``. When ``BOp=None``, this can be the true solution or the data projected in the
        model space (note that ``mask`` will be still applied). When ``BOp`` is provided,
    mask : :obj:`numpy.ndarray`
        Mask of size  ``ny`` (subsampling along one axis) or ``(ny, nx, nt)`` (subsampling along two axes).
        When using BOp simply provide a mask of ones with the size of the reconstructed data
    samplings : :obj:`tuple`
        Sampling intervals for y, x, and t axes
    nfft : :obj:`int`
        Number of samples of Fourier transform (currently the same value will be use for all dimensions)
    thresh : :obj:`float` or :obj:`numpy.ndarray`
        Threshold parameters. When `None` is provide, the parameters are inferred directly from the data
    threshkind : :obj:`str`, optional
        Threshold kind: linear, exponential, or exponential1 (see :mod:`pypocs.threshold` for details)
    BOp : :obj:`pylops.LinearOperator`, optional
        Bilinear interpolation to use instead of the restriction operator for off-the-grid receivers.
        If passed, mask will be only used to infer the dimensions of the reconstructed model
    MOp : :obj:`pylops.LinearOperator`, optional
        Frequency-wavenumber masking operator (if passed, it will be chained to the frequency-wavenumber sparsifying
        transform)
    niter : :obj:`int`, optional
        Number of iterations
    xtrue : :obj:`numpy.ndarray`, optional
        True solution of size ``(nx, nt)``. This must be provided when interested to obtain an
         error curve as function of iterations
    masktrue : :obj:`numpy.ndarray`, optional
        Mask of size ``(nx, nt)`` to apply to the true and estimated solution in the computation of the
         error curve as function of iterations
    history : :obj:`bool`, optional
        Deprecated
    backend : :obj:`str`, optional
        Backend (``numpy`` or ``cupy``)
    verb : :obj:`bool`, optional
        Verbose iterations

    Returns
    -------
    xrec : :obj:`numpy.ndarray`
        Reconstructed data of size ``(nx, nt)``
    err : :obj:`numpy.ndarray`
        Error norm as function of iterations
    xhistpocs : :obj:`list`, optional
        History of solutions (only if ``history>0``)

    """
    ncp = get_module(backend)
    dx, dx, dt = samplings
    if BOp is None:
        ny, nx, nt = x.shape
    else:
        ny, nx, nt = mask.shape

    # History (for backward compatibility with bool)
    history = -1 if isinstance(history, bool) else history

    # Convert to cupy
    if backend == "cupy":
        x = ncp.asarray(x)
        mask = ncp.asarray(mask)
        if xtrue is not None:
            xtrue = ncp.asarray(xtrue)
        if masktrue is not None:
            masktrue = ncp.asarray(masktrue)

    # Scale to -1-1
    scaling = np.max(np.abs(x))
    x = x / scaling

    # Create mask
    if BOp is None and mask.ndim == 1:
        # subsampling over one spatial axis
        iava = np.where(mask==1)[0]
        mask = ncp.zeros((ny, nx, nt))
        mask[iava] = 1
    
    # Create FFT
    FFTOp = FFTND(dims=[ny, nx, nt], nffts=[nfft, nfft, nfft], sampling=[dx, dx, dt], real=True,
                  engine="numpy" if backend == "cupy" else "scipy")
    FFTOp.nffts = list(FFTOp.nffts)

    # Create data
    if BOp is None:
        y = mask * x
        ymask = y
    else:
        y = x
        if thresh is None:
            ymask = BOp.H @ y

    # Create sparsifying transform
    if MOp is None:
        SOp = FFTOp.H
    else:
        SOp = FFTOp.H * MOp

    # Threshold (if not already provided)
    if thresh is None:
        # Automatically identified from the spectrum of the subsampled
        pmax, pmin = 0.99, 0.01 # percentage of max
        ymax = float(np.max(np.abs(FFTOp * ymask.ravel())))
        thresh = [pmin * ymax, pmax * ymax]

    if threshkind == 'linear':
        thresh = threshold.linear(thresh[1], thresh[0], niter)
    elif threshkind == 'exponential':
        thresh = threshold.exponential(thresh[1], thresh[0], niter)
    elif threshkind == 'exponential1':
        thresh = threshold.exponential1(thresh[1], thresh[0], niter)
    if verb:
        print(f'threshkind={threshkind}')
        print(f'thesh={thresh}')

    # POCS
    xrec, xhist, _, err, _ = \
        POCS(y, mask, SOp, (nfft, nfft, nfft), thresh, niter, BOp=BOp,
             dinit=xinit, dtrue=None if xtrue is None else xtrue / scaling,
             masktrue=masktrue, history=history)
    
    # Rescale back
    xrec = xrec * scaling

    # Convert back to numpy
    if backend == "cupy":
        xrec = ncp.asnumpy(xrec)
        if xtrue is not None:
            err = ncp.asnumpy(err)

    if history == -1:
        return xrec, err
    else:
        xhistpocs = []
        for i in range(len(xhist)):
            if backend == "cupy":
                xhistpocs.append(ncp.asnumpy(xhist[i]) * float(scaling))
            else:
                xhistpocs.append(xhist[i] * float(scaling))
        return xrec, err, xhistpocs


def hqs_interpolate(x, mask, samplings, nfft, thresh, niter=10,
                    affine=True, identityop=True, norm=L0,
                    BOp=None, BOpiters=1, MOp=None,
                    xtrue=None, masktrue=None,
                    history=-1, backend="numpy", verb=False):
    """HQS interpolator

    Wrapper function to run interpolation with :func:`pyproximal.optimization.primal.HQS`. This
    interpolator is equivalent to the POCS interpolator but behaves in a similar fashion to the other pyproximal's
    powered interpolator and should be preferred over :func:`pypocs.interpolation.pocs_interpolate`

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Data of size ``(nx, nt)``. This could be the true solution or the data projected in the model space
        (note that ``mask`` will be still applied)
    mask : :obj:`numpy.ndarray`
        Mask of size  ``ny`` (subsampling along one axis) or ``(ny, nx, nt)`` (subsampling along two axes)
    samplings : :obj:`tuple`
        Sampling intervals for y, x, and t axes
    nfft : :obj:`int`
        Number of samples of Fourier transform (currently the same value will be use for all dimensions)
    thresh : :obj:`list`
        Threshold parameters. When a single parameters is provided, it uses constant threshold;
        when two parameters are provided, it uses exponential threshold (equivalent to
         :func:`pypocs.threshold.exponential1`)
    niter : :obj:`int`, optional
        Number of iterations
    affine : :obj:`bool`, optional
        Deprecated
    identityop : :obj:`bool`, optional
        Deprecated
    norm : :obj:`pyproximal.proximal`, optional
        Norm to use for regularization term
    BOp : :obj:`pylops.LinearOperator`, optional
        Bilinear interpolation to use instead of the restriction operator for off-the-grid receivers.
        If passed, mask will be only used to infer the dimensions of the reconstructed model
    BOpiters : :obj:`int`, optional
        Number of LSQR iterations to evaluate the proximal operator of the Affine set when ``BOp`` is used instead of
        the restriction operator
    MOp : :obj:`pylops.LinearOperator`, optional
        Frequency-wavenumber masking operator (if passed, it will be chained to the frequency-wavenumber sparsifying
        transform)
    xtrue : :obj:`numpy.ndarray`, optional
        True solution of size ``(nx, nt)``. This must be provided when interested to obtain an
         error curve as function of iterations
    masktrue : :obj:`numpy.ndarray`, optional
        Mask of size ``(nx, nt)`` to apply to the true and estimated solution in the computation of the
         error curve as function of iterations
    history : :obj:`int`, optional
        Number of steps after which the current solution is collected (use ``-1`` to avoid collecting it)
    backend : :obj:`str`, optional
        Backend (``numpy`` or ``cupy``)
    verb : :obj:`bool`, optional
        Verbose iterations

    Returns
    -------
    xrec : :obj:`numpy.ndarray`
        Reconstructed data of size ``(nx, nt)``
    errhqs : :obj:`numpy.ndarray`
        Error norm as function of iterations
    xhisthqs : :obj:`list`, optional
        History of solutions (only if ``history>0``)

    """
    ncp = get_module(backend)
    dx, dx, dt = samplings

    if BOp is None:
        ny, nx, nt = x.shape
        xsize = x.size
    else:
        ny, nx, nt = mask.shape
        xsize = mask.size

    # History (for backward compatibility with bool)
    history = -1 if isinstance(history, bool) else history

    # Convert to cupy
    if backend == "cupy":
        x = ncp.asarray(x)
        if xtrue is not None:
            xtrue = ncp.asarray(xtrue)

    # Scale to -1-1
    scaling = np.max(np.abs(x))
    x = x / scaling

     # Create mask
    if mask.ndim == 1:
        # subsampling over one spatial axis
        iava = np.where(mask==1)[0]
        mask = ncp.zeros((ny, nx, nt))
        mask[iava] = 1

        # Create restriction operator
        Rop = Restriction(dims=(x.shape[0], x.shape[1], x.shape[2]), iava=iava, axis=0, dtype='float64')

    else:
        # subsampling over two spatial axes
        iava = np.where(mask.ravel()==1)[0]

        # Create restriction operator
        Rop = Restriction(dims=xsize, iava=iava, dtype='float64')

    # Create data
    if BOp is None:
        y = Rop @ x.ravel()
        if thresh is None:
            ymask = Rop.H @ y
    else:
        y = x
        if thresh is None:
            ymask = BOp.H @ y

    # Create FFT
    FFTop = FFTND(dims=[ny, nx, nt], nffts=[nfft, nfft, nfft], sampling=[dx, dx, dt], real=True,
                  engine="numpy" if backend == "cupy" else "scipy")
    FFTop.nffts = list(FFTop.nffts)

    # Create sparsifying transform
    if MOp is None:
        SOp = FFTop
    else:
        SOp = MOp * FFTop

    # Callback
    if xtrue is not None:
        callback = Callback(xtrue.ravel() / scaling, history=history, masktrue=masktrue, backend=backend)
        cb = lambda xx: callback(xx)
    else:
        cb = None

    # Choose thresholding strategy
    if thresh is None:
        # Automatically identified from the spectrum of the subsampled
        pmax, pmin = 0.99, 0.01  # percentage of max
        ymax = float(np.max(np.abs(FFTop * ymask.ravel())))
        thresh = [pmin * ymax, pmax * ymax]

    if len(thresh) == 1:
        #constant
        sigmaiters = thresh[0] * np.ones(niter)
        sigma = thresh[0]
    else:
        sigmamax = thresh[1]
        sigmamin = thresh[0]
        sigmaiters = threshold.exponential1(sigmamax, sigmamin, niter)
        sigmaiters = np.insert(sigmaiters, 0, 0)
        sigma = lambda x: sigmaiters[x]

    if verb:
        print(f'thesh={sigmaiters[1:]}')

    # Define problem
    if affine and identityop:
        if BOp is None:
            laff = AffineSet(Rop, y.ravel(), niter=1)
        else:
            laff = AffineSet(BOp, y.ravel(), niter=BOpiters)
        lort = Orthogonal(norm(sigma), SOp)
        x0 = Rop.H * y.ravel()
    else:
        # Raise error that user is trying to do something else than POCS...
        raise NotImplementedError('POCS requires affine=True and identityop=True')

    with pylops.config.disabled_ndarray_multiplication():
        if verb:
            print('HQS interpolation')
            print(f'f={laff if affine else lort}')
            print(f'g={lort if affine else laff}')
        xrec = HQS(laff, lort, x0=x0, tau=1., niter=niter, show=False,
                   callback=cb)[0]
    xrec = xrec.reshape(x.shape)

    # Rescale back
    xrec = xrec * scaling

    # Convert back to numpy
    if backend == "cupy":
        xrec = ncp.asnumpy(xrec)

    errhqs = None
    if xtrue is not None:
        errhqs = np.array(callback.err)

    if history == -1:
        return xrec, errhqs
    else:
        xhisthqs = []
        for i in range(len(callback.xhist)):
            xhisthqs.append(callback.xhist[i] * float(scaling))
        return xrec, errhqs, xhisthqs


def admm_interpolate(x, mask, samplings, nfft, thresh, niter=10,
                     affine=True, identityop=True, norm=L0,
                     BOp=None, BOpiters=1, MOp=None,
                     xtrue=None, masktrue=None,
                     history=-1, backend="numpy", verb=False):
    """ADMM interpolator

    Wrapper function to run interpolation with :func:`pyproximal.optimization.primal.ADMM` or
    :func:`pyproximal.optimization.primal.LinearizedADMM`.

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Data of size ``(nx, nt)``. This could be the true solution or the data projected in the model space
        (note that ``mask`` will be still applied)
    mask : :obj:`numpy.ndarray`
        Mask of size  ``ny`` (subsampling along one axis) or ``(ny, nx, nt)`` (subsampling along two axes)
    samplings : :obj:`tuple`
        Sampling intervals for y, x, and t axes
    nfft : :obj:`int`
        Number of samples of Fourier transform (currently the same value will be use for all dimensions)
    thresh : :obj:`list`
        Threshold parameters. When a single parameters is provided, it uses constant threshold;
        when two parameters are provided, it uses exponential threshold (equivalent to
         :func:`pypocs.threshold.exponential1`)
    niter : :obj:`int`, optional
        Number of iterations
    affine : :obj:`bool`, optional
        Use AffineSet (``True``) or Euclidean Ball (``True``) for data term
    identityop : :obj:`bool`, optional
        Use ADMM (``True``) or L-ADMM (``False``). In the case of ADMM the K linear operator is chosen to the
        the Fourier sparsifying transform for ``Affine=True`` or the restriction operator ``Affine=False``
    norm : :obj:`pyproximal.proximal`, optional
        Norm to use for regularization term
    BOp : :obj:`pylops.LinearOperator`, optional
        Bilinear interpolation to use instead of the restriction operator for off-the-grid receivers.
        If passed, mask will be only used to infer the dimensions of the reconstructed model
    BOpiters : :obj:`int`, optional
        Number of LSQR iterations to evaluate the proximal operator of the Affine set when ``BOp`` is used instead of
        the restriction operator
    MOp : :obj:`pylops.LinearOperator`, optional
        Frequency-wavenumber masking operator (if passed, it will be chained to the frequency-wavenumber sparsifying
        transform)
    xtrue : :obj:`numpy.ndarray`, optional
        True solution of size ``(nx, nt)``. This must be provided when interested to obtain an
         error curve as function of iterations
    masktrue : :obj:`numpy.ndarray`, optional
        Mask of size ``(nx, nt)`` to apply to the true and estimated solution in the computation of the
         error curve as function of iterations
    history : :obj:`int`, optional
        Number of steps after which the current solution is collected (use ``-1`` to avoid collecting it)
    backend : :obj:`str`, optional
        Backend (``numpy`` or ``cupy``)
    verb : :obj:`bool`, optional
        Verbose iterations

    Returns
    -------
    xrec : :obj:`numpy.ndarray`
        Reconstructed data of size ``(nx, nt)``
    errhqs : :obj:`numpy.ndarray`
        Error norm as function of iterations
    xhisthqs : :obj:`list`, optional
        History of solutions (only if ``history>0``)

    """
    ncp = get_module(backend)
    dx, dx, dt = samplings

    if BOp is None:
        ny, nx, nt = x.shape
        xsize = x.size
    else:
        ny, nx, nt = mask.shape
        xsize = mask.size

    # History (for backward compatibility with bool)
    history = -1 if isinstance(history, bool) else history

    # Convert to cupy
    if backend == "cupy":
        x = ncp.asarray(x)
        # mask = ncp.asarray(mask)
        if xtrue is not None:
            xtrue = ncp.asarray(xtrue)

    # Scale to -1-1
    scaling = np.max(np.abs(x))
    x = x / scaling

    # Create mask
    if mask.ndim == 1:
        # subsampling over one spatial axis
        iava = np.where(mask == 1)[0]
        mask = ncp.zeros((ny, nx, nt))
        mask[iava] = 1

        # Create restriction operator
        Rop = Restriction(dims=(x.shape[0], x.shape[1], x.shape[2]), iava=iava, axis=0, dtype='float64')

    else:
        # subsampling over two spatial axes
        iava = np.where(mask.ravel() == 1)[0]

        # Create restriction operator
        Rop = Restriction(dims=xsize, iava=iava, dtype='float64')

    # Create data
    if BOp is None:
        y = Rop @ x.ravel()
        if thresh is None:
            ymask = Rop.H @ y
    else:
        y = x
        if thresh is None:
            ymask = BOp.H @ y

    # Create FFT
    FFTop = FFTND(dims=[ny, nx, nt], nffts=[nfft, nfft, nfft], sampling=[dx, dx, dt], real=True,
                  engine="numpy" if backend == "cupy" else "scipy")
    FFTop.nffts = list(FFTop.nffts)

    # Create sparsifying transform
    if MOp is None:
        SOp = FFTop
    else:
        SOp = MOp * FFTop

    # Callback
    if xtrue is not None:
        callback = Callback(xtrue.ravel() / scaling, history=history, masktrue=masktrue, backend=backend)
        cb = lambda xx: callback(xx)
    else:
        cb = None

    # Choose thresholding strategy
    if thresh is None:
        # Automatically identified from the spectrum of the subsampled
        pmax, pmin = 0.99, 0.01  # percentage of max
        ymax = float(np.max(np.abs(FFTop * ymask.ravel())))
        thresh = [pmin * ymax, pmax * ymax]

    if len(thresh) == 1:
        # constant
        sigmaiters = thresh[0] * np.ones(niter)
        sigma = thresh[0]
    else:
        sigmamax = thresh[1]
        sigmamin = thresh[0]
        sigmaiters = threshold.exponential1(sigmamax, sigmamin, niter)
        sigmaiters = np.insert(sigmaiters, 0, 0)
        sigma = lambda x: sigmaiters[x]

    if verb:
        print(f'thesh={sigmaiters[1:]}')

    # Define problem
    if affine and identityop:
        if BOp is None:
            laff = AffineSet(Rop, y.ravel(), niter=1)
        else:
            laff = AffineSet(BOp, y.ravel(), niter=BOpiters)
        lort = Orthogonal(norm(sigma), SOp)
        K = Identity(xsize)
        tau, mu = .99, .99
        x0 = Rop.H * y.ravel()
    elif affine and not identityop:
        if BOp is None:
            laff = AffineSet(Rop, y.ravel(), niter=1)
        else:
            laff = AffineSet(BOp, y.ravel(), niter=BOpiters)
        lort = norm(sigma)
        K = SOp
        tau, mu = .99, .99
        x0 = Rop.H * y.ravel()
    elif not affine:
        laff = EuclideanBall(y.ravel(), radius=1e-10)
        lort = Orthogonal(norm(sigma), SOp)
        K = Rop
        tau, mu = .9, .9
        x0 = ncp.zeros_like(x).ravel()

    with pylops.config.disabled_ndarray_multiplication():
        if verb:
            print('ADMM interpolation')
            print(f'f={laff if affine else lort}')
            print(f'g={lort if affine else laff}')
            print(f'K={K}')
            print(f'tau={tau}, mu={mu}')
        if affine and identityop:
            if verb: print('solver=ADMM')
            xrec = ADMM(laff, lort,
                        x0=x0, tau=tau, niter=niter, gfirst=True, show=False,
                        callback=lambda xx: callback(xx))[0]
        else:
            if verb: print('solver=LinearizedADMM')
            xrec = LinearizedADMM(laff if affine else lort, lort if affine else laff, K,
                                  x0=x0, tau=tau, mu=mu, niter=niter, show=False,
                                  callback=cb)[0]
    xrec = xrec.reshape(x.shape)

    # Rescale back
    xrec = xrec * scaling

    # Convert back to numpy
    if backend == "cupy":
        xrec = ncp.asnumpy(xrec)

    erradmm = None
    if xtrue is not None:
        errpd = np.array(callback.err)

    if history == -1:
        return xrec, erradmm
    else:
        xhistadmm = []
        for i in range(len(callback.xhist)):
            xhistadmm.append(callback.xhist[i] * float(scaling))
        return xrec, erradmm, xhistadmm


def pd_interpolate(x, mask, samplings, nfft, thresh,
                   threshkind='constant', niter=10,
                   affine=True, identityop=True, norm=L0,
                   BOp=None, BOpiters=1, MOp=None, xinit=None,
                   xtrue=None, masktrue=None,
                   history=-1, backend="numpy", verb=False):
    """PD interpolator

    Wrapper function to run interpolation with :func:`pyproximal.optimization.primaldual.PrimalDual`

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        Data of size ``(nx, nt)``. This could be the true solution or the data projected in the model space
        (note that ``mask`` will be still applied)
    mask : :obj:`numpy.ndarray`
        Mask of size  ``ny`` (subsampling along one axis) or ``(ny, nx, nt)`` (subsampling along two axes)
    samplings : :obj:`tuple`
        Sampling intervals for y, x, and t axes
    nfft : :obj:`int`
        Number of samples of Fourier transform (currently the same value will be use for all dimensions)
    thresh : :obj:`list`
        Threshold parameters. When a single parameters is provided, it uses constant threshold;
        when two parameters are provided, it uses exponential threshold (equivalent to
         :func:`pypocs.threshold.exponential1`). When `None` is provide, the parameters are
          inferred directly from the data
    threshkind : :obj:`str`, optional
        Threshold kind: constant or exponential1 (see :mod:`pypocs.threshold` for details)
    niter : :obj:`int`, optional
        Number of iterations
    affine : :obj:`bool`, optional
        Use AffineSet (``True``) or Euclidean Ball (``True``) for data term
    identityop : :obj:`bool`, optional
        Use ADMM (``True``) or L-ADMM (``False``). In the case of ADMM the K linear operator is chosen to the
        the Fourier sparsifying transform for ``Affine=True`` or the restriction operator ``Affine=False``
    norm : :obj:`pyproximal.proximal`, optional
        Norm to use for regularization term
    BOp : :obj:`pylops.LinearOperator`, optional
        Bilinear interpolation to use instead of the restriction operator for off-the-grid receivers.
        If passed, mask will be only used to infer the dimensions of the reconstructed model
    BOpiters : :obj:`int`, optional
        Number of LSQR iterations to evaluate the proximal operator of the Affine set when ``BOp`` is used instead of
        the restriction operator
    MOp : :obj:`pylops.LinearOperator`, optional
        Frequency-wavenumber masking operator (if passed, it will be chained to the frequency-wavenumber sparsifying
        transform)
    xtrue : :obj:`numpy.ndarray`, optional
        True solution of size ``(nx, nt)``. This must be provided when interested to obtain an
         error curve as function of iterations
    masktrue : :obj:`numpy.ndarray`, optional
        Mask of size ``(nx, nt)`` to apply to the true and estimated solution in the computation of the
         error curve as function of iterations
    history : :obj:`int`, optional
        Number of steps after which the current solution is collected (use ``-1`` to avoid collecting it)
    backend : :obj:`str`, optional
        Backend (``numpy`` or ``cupy``)
    verb : :obj:`bool`, optional
        Verbose iterations

    Returns
    -------
    xrec : :obj:`numpy.ndarray`
        Reconstructed data of size ``(nx, nt)``
    errhqs : :obj:`numpy.ndarray`
        Error norm as function of iterations
    xhisthqs : :obj:`list`, optional
        History of solutions (only if ``history>0``)

    """
    ncp = get_module(backend)
    dx, dx, dt = samplings

    if BOp is None:
        ny, nx, nt = x.shape
        xsize = x.size
    else:
        ny, nx, nt = mask.shape
        xsize = mask.size

    # History (for backward compatibility with bool)
    history = -1 if isinstance(history, bool) else history

    # Convert to cupy
    if backend == "cupy":
        x = ncp.asarray(x)
        if xtrue is not None:
            xtrue = ncp.asarray(xtrue)

    # Scale to -1-1
    scaling = np.max(np.abs(x))
    x = x / scaling
       
     # Create mask
    if mask.ndim == 1:
        # subsampling over one spatial axis
        iava = np.where(mask==1)[0]
        mask = ncp.zeros((ny, nx, nt))
        mask[iava] = 1

        # Create restriction operator
        Rop = Restriction(dims=(x.shape[0], x.shape[1], x.shape[2]), iava=iava, axis=0, dtype='float64')

    else:
        # subsampling over two spatial axes
        iava = np.where(mask.ravel()==1)[0]

        # Create restriction operator
        Rop = Restriction(dims=xsize, iava=iava, dtype='float64')
    
    # Create data
    if BOp is None:
        y = Rop @ x.ravel()
        if thresh is None:
            ymask = Rop.H @ y
    else:
        y = x
        if thresh is None:
            ymask = BOp.H @ y

    # Create FFT
    FFTop = FFTND(dims=[ny, nx, nt], nffts=[nfft, nfft, nfft], sampling=[dx, dx, dt], real=True,
                  engine="numpy" if backend == "cupy" else "scipy")
    FFTop.nffts = list(FFTop.nffts)

    # Create sparsifying transform
    if MOp is None:
        SOp = FFTop
    else:
        SOp = MOp * FFTop

    # Callback
    if xtrue is not None:
        callback = Callback(xtrue.ravel() / scaling, history=history, masktrue=masktrue, backend=backend)
        cb = lambda xx: callback(xx)
    else:
        cb = None

    # Choose thresholding strategy
    if thresh is None:
        # Automatically identified from the spectrum of the subsampled
        pmax, pmin = 0.99, 0.01  # percentage of max
        ymax = float(np.max(np.abs(FFTop * ymask.ravel())))
        thresh = [pmin * ymax, pmax * ymax]
        if threshkind == 'constant':
            thresh = [(thresh[0] + thresh[1]) / 2., ] # use middle point

    if len(thresh) == 1:
        #constant
        sigmaiters = thresh[0] * np.ones(niter)
        sigma = thresh[0]
    else:
        sigmamax=thresh[1]
        sigmamin=thresh[0]
        sigmaiters = threshold.exponential1(sigmamax, sigmamin, niter)
        sigmaiters = np.insert(sigmaiters, 0, 0)
        sigma = lambda x: sigmaiters[x]

    if verb:
        print(f'thesh={sigmaiters[1:]}')

    # Define problem
    if affine and identityop:
        if BOp is None:
            laff = AffineSet(Rop, y.ravel(), niter=1)
        else:
            laff = AffineSet(BOp, y.ravel(), niter=BOpiters)
        lort = Orthogonal(norm(sigma), SOp)
        K = Identity(xsize)
        tau, mu = .99, .99
        if xinit is None:
            xinit = Rop.H * y.ravel()
    elif affine and not identityop:
        if BOp is None:
            laff = AffineSet(Rop, y.ravel(), niter=1)
        else:
            laff = AffineSet(BOp, y.ravel(), niter=BOpiters)
        lort = norm(sigma)
        K = SOp
        tau, mu = .99, .99
        if xinit is None:
            xinit = Rop.H * y.ravel()
    elif not affine:
        laff = EuclideanBall(y.ravel(), radius=1e-10)
        lort = Orthogonal(norm(sigma), SOp)
        K = BOp if BOp is not None else Rop
        tau, mu = .9, .9
        if xinit is None:
            xinit = ncp.zeros_like(x).ravel()

    with pylops.config.disabled_ndarray_multiplication():
        if verb:
            print('PD interpolation')
            print(f'f={laff if affine else lort}')
            print(f'g={lort if affine else laff}')
            print(f'K={K}')
            print(f'tau={tau}, mu={mu}')
        xrec = PrimalDual(laff if affine else lort, lort if affine else laff, K,
                          x0=xinit, tau=tau, mu=mu, niter=niter, gfirst=True, show=False,
                          callback=cb)
    xrec = xrec.reshape(ny, nx, nt)
    
    # Rescale back
    xrec = xrec * scaling

    # Convert back to numpy
    if backend == "cupy":
        xrec = ncp.asnumpy(xrec)

    errpd = None
    if xtrue is not None:
        errpd = np.array(callback.err)

    if history == -1:
        return xrec, errpd
    else:
        xhistpd = []
        for i in range(len(callback.xhist)):
            xhistpd.append(callback.xhist[i] * float(scaling))
        return xrec, errpd, xhistpd
