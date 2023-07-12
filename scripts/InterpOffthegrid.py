#!/usr/bin/env python
# coding: utf-8

import argparse
import yaml

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.io as sp_io
import pylops
import cupy as cp
cp_asarray = cp.asarray
cp_asnumpy = cp.asnumpy

from pylops.utils import dottest
from pylops.utils.wavelets import *
from pylops.utils.seismicevents import *
from pylops.basicoperators import *
from pylops.signalprocessing import *
from pylops.waveeqprocessing import *
from pylops.optimization.leastsquares import *
from pylops.signalprocessing.patch3d import patch3d_design

from pyproximal.proximal import *
from pyproximal.optimization.primal import *
from pyproximal.optimization.primaldual import *

from pypocs import threshold
from pypocs.visual import explode_volume
from pypocs.POCS import POCS
from pypocs.metric import *
from pypocs.interpolation import Callback


def main(parser):

    ######### INPUT PARAMS #########
    parser.add_argument('-c', '--config', type=str, help='Configuration file')
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        setup = yaml.load(stream, Loader=yaml.FullLoader)

    filename = setup['exp']['inputfile']
    outfilename = setup['exp']['outputfile']
    plotflag = setup['exp']['plotflag']

    dry = setup['data']['dry']
    drx = setup['data']['drx']
    dt = setup['data']['dt']

    perc_subsampling = setup['subsampling']['perc_subsampling']
    seed = setup['subsampling']['seed']

    threshmask = setup['preprocessing']['threshmask']
    itoff = setup['preprocessing']['itoff']
    vshift = setup['preprocessing']['vshift']
    nwin = setup['preprocessing']['nwin']
    nover = setup['preprocessing']['nover']
    nop = setup['preprocessing']['nop']
    epsweighting = setup['preprocessing']['epsweighting']
    sigmaweighting = setup['preprocessing']['sigmaweighting']
    threshweighting = setup['preprocessing']['threshweighting']
    scweighting = setup['preprocessing']['scweighting']
    nfft = setup['preprocessing']['nfft']

    niter = setup['interpolation']['niter']
    niteraffine = setup['interpolation']['niteraffine']
    thresh = setup['interpolation']['thresh']
    tau = setup['interpolation']['tau']
    mu = setup['interpolation']['mu']
    jsrnsave = setup['interpolation']['jsrnsave']

    # Display experiment setup
    sections = ['exp', 'data', 'subsampling',
                'preprocessing', 'interpolation']
    print('---------------------------------------------------------')
    print('POCS Interpolation of Off-the-grid acquisition geometry')
    print('---------------------------------------------------------\n')
    for section in sections:
        print(section.upper())
        for key, value in setup[section].items():
            print(f'{key} = {value}')
        print('\n-------------------------------\n')

    np.random.seed(seed)

    ######### Data loading and preprocessing #########

    fdata = np.load(filename)

    # Load data
    data = fdata['data']
    nrx, nry, nt = data.shape
    t = fdata['t']

    # Normalize data
    data = data / np.max(np.abs(data))

    # Acquisition
    RECX = fdata['RECX']
    RECY = fdata['RECY']
    recz = fdata['recz']
    SRCX = fdata['SRCX']
    SRCY = fdata['SRCY']
    srcz = fdata['srcz']

    # Subsampling locations
    Nsub = int(np.round(nrx * perc_subsampling))
    iy = np.arange(nrx)
    ix = np.arange(nry)

    iygrid, ixgrid = np.meshgrid(iy, ix, indexing='ij')

    iyava_y = np.random.randint(2, nrx - 2, Nsub)
    iyava = np.tile(iyava_y, (nry, 1)).ravel()
    ixava = np.tile(np.arange(nry), (Nsub, 1)).T.ravel()

    mask2d = np.zeros((nrx, nry))
    mask2d[iyava, ixava] = 1
    iava2d = np.where(mask2d.ravel() == 1)[0]

    mask3d = np.zeros((nrx, nry, nt))
    mask3d[iyava, ixava, :] = 1
    iava3d = np.where(mask3d.ravel() == 1)[0]

    iyava = iygrid.ravel()[iava2d]
    ixava = ixgrid.ravel()[iava2d]
    Nsub = int(len(iava3d) / nt)

    # Create restriction operator
    Rop = Restriction(nrx * nry * nt, iava=iava3d.ravel(), dtype='float64')

    # Create bilinear operator
    iyava_pert = iyava + np.random.uniform(-0.5, 0.5, Nsub)
    ixava_pert = ixava + np.random.uniform(-0.5, 0.5, Nsub)
    ixava_pert[ixava_pert < 0] = 0.
    ixava_pert[ixava_pert > nry - 2] = nry - 2
    iava3d1 = np.vstack((iyava_pert, ixava_pert))

    Bop = Bilinear(cp_asarray(iava3d1), (nrx, nry, nt), dtype='float64')
    print(f'Restriction operator {Rop}, Bilinear Operator: {Bop}')

    # Off-grid data
    datasub = Bop @ cp_asarray(data)
    datamasked = Bop.H @ datasub
    datamasked = cp_asnumpy(datamasked)

    # FK spectra
    FFTop = FFTND(dims=[nrx, nry, nt], nffts=[nfft, nfft, nfft], sampling=[drx, dry, dt], real=True)

    datafk = FFTop @ data
    datamaskedfk = FFTop @ datamasked

    # Create direct arrival mask
    direct_mask = np.ones_like(data)
    direct_thresh = threshmask * np.max(np.abs(data))
    for iy in range(nrx):
        for ix in range(nry):
            it = np.where(np.abs(data[iy, ix]) > direct_thresh)[0]
            if len(it) > 0:
                direct_mask[iy, ix, :max(0, it[0] - itoff)] = 0.

    # Patched fk sparsyfing transform
    dimsd = data.shape
    nop1 = (nop[0], nop[1], nop[2] // 2 + 1)

    nwins, dims, mwins_inends, dwins_inends = patch3d_design(dimsd, nwin, nover, nop1)

    F1op = FFTND(nwin, nffts=nop, real=True, dtype="complex64")
    Srecop = Patch3D(F1op.H, dims, dimsd, nwin, nover, nop1, tapertype='cosine')

    # Shift operator
    shift = np.sqrt((recz - srcz) ** 2 +
                    (RECX.reshape(nry, nrx) - SRCX) ** 2 +
                    (RECY.reshape(nry, nrx) - SRCY) ** 2) / vshift

    Shiftop = Shift((nrx, nry, nt), shift=-shift.T, sampling=dt, axis=2)

    # Shift data
    datashifted = np.real(Shiftop * data)

    # Shift subsampled data
    datashiftedsub = Bop @ cp_asarray(datashifted)
    datashiftedmasked = Bop.H @ datashiftedsub
    datashiftedmasked = cp_asnumpy(datashiftedmasked)

    datashiftedfk = FFTop @ datashifted
    datashiftedmaskedfk = FFTop @ datashiftedmasked

    # Find normalization factors
    weightshifted = 1. / (sp.ndimage.gaussian_filter(np.abs(sp.signal.hilbert(datashifted)), sigma=sigmaweighting) + epsweighting)
    weightshifted[weightshifted > threshweighting] = threshweighting  # remove infinite values when there is no signal
    weightshifted /= (weightshifted.max() / scweighting)

    # Apply weighting with subsampled data
    datashiftedsub = Bop * cp_asarray(datashifted * weightshifted)
    datashiftedmasked = Bop.H * datashiftedsub

    datashiftedsub = cp_asnumpy(datashiftedsub)
    datashiftedmasked = cp_asnumpy(datashiftedmasked)

    ######### Interpolation #########
    laff = AffineSet(Bop, cp.asarray(datashiftedsub).ravel(), niter=niteraffine)
    lort = L0(thresh)
    callback = Callback(cp_asarray(datashifted * weightshifted).ravel(), history=jsrnsave,
                        masktrue=cp_asarray(direct_mask / weightshifted).ravel(), backend="cupy")
    cb = lambda xx: callback(xx)

    xpd = \
        PrimalDual(laff, lort, Srecop.H, x0=Bop.H * cp_asarray(datashiftedsub.ravel()),
                   tau=tau, mu=mu, niter=niter, gfirst=True, show=False, callback=cb)
    errpd = [np.linalg.norm(datashifted - datashiftedmasked / weightshifted) / np.linalg.norm(
        datashifted)] + callback.err
    snr_hist_pd = 20 * np.log10(1 / np.array(errpd))

    # Renormalize and shift back reconstructed data
    datarec_pdshiftedback = cp_asnumpy(np.real(Shiftop.H * (xpd.reshape(nrx, nry, nt) / cp_asarray(weightshifted))))

    # Final SNR
    _ = metrics(data, direct_mask*datarec_pdshiftedback, verb=True)

    ######### Saving results #########
    if outfilename is not None:
        np.savez(outfilename, data=direct_mask * datarec_pdshiftedback, t=t,
                 RECX=RECX, RECY=RECY, SRCX=SRCX, SRCY=SRCY,
                 snr_hist=snr_hist_pd[::jsrnsave])

    ######### Plotting #########
    if plotflag:
        # Geometry
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.scatter(iygrid, ixgrid, c='k')
        ax.scatter(iyava, ixava, c='r')
        ax.scatter(iava3d1[0], iava3d1[1], c='b')
        ax.set_title('Geometry')

        # Data (full and subsampled)
        explode_volume(data.transpose(2, 0, 1), x=30, clipval=(-0.2, 0.2), figsize=(8, 8), title='Full data')
        explode_volume(datamasked.transpose(2, 0, 1), x=30, clipval=(-0.2, 0.2), figsize=(8, 8),
                       title='Subsampled data')

        # FK spectra
        explode_volume(np.fft.fftshift(np.abs(datafk[..., :100]), axes=(0, 1)).transpose(2, 0, 1), t=30,
                       tlim=[0, FFTop.fs[-1][100]], tlabel='f(Hz)',
                       cmap='jet', figsize=(8, 8), clipval=(0, 0.2))

        explode_volume(np.fft.fftshift(np.abs(datamaskedfk[..., :100]), axes=(0, 1)).transpose(2, 0, 1), t=30,
                       tlim=[0, FFTop.fs[-1][100]], tlabel='f(Hz)',
                       cmap='jet', figsize=(8, 8), clipval=(0, 0.2))

        # Direct wave mask
        explode_volume(direct_mask.transpose(2, 0, 1), x=30, cmap='gray_r', clipval=(0, 1), figsize=(8, 8),
                       title='Direct Mask')
        explode_volume((direct_mask * data).transpose(2, 0, 1), x=30, clipval=(-0.2, 0.2), figsize=(8, 8),
                       title='Masked data')

        # Data Shift
        plt.figure(figsize=(8, 3))
        plt.imshow(shift)
        plt.axis('tight')
        plt.colorbar()
        plt.title('Shift')

        explode_volume(datashifted.transpose(2, 0, 1), x=30, clipval=(-0.2, 0.2), figsize=(8, 8),
                       title='Shifted data')

        # FK spectra of shifted data
        explode_volume(np.fft.fftshift(np.abs(datashiftedfk[..., :100]), axes=(0, 1)).transpose(2, 0, 1), t=30,
                       tlim=[0, FFTop.fs[-1][100]], tlabel='f(Hz)',
                       cmap='jet', figsize=(8, 8), clipval=(0, 0.2))

        explode_volume(np.fft.fftshift(np.abs(datashiftedmaskedfk[..., :100]), axes=(0, 1)).transpose(2, 0, 1), t=30,
                       tlim=[0, FFTop.fs[-1][100]], tlabel='f(Hz)',
                       cmap='jet', figsize=(8, 8), clipval=(0, 0.2))

        # Data weighting
        explode_volume(weightshifted.transpose(2, 0, 1), x=30, figsize=(8, 8), clipval=(0, 10),
                       cmap='jet', title='Data weight')

        explode_volume((datashifted * weightshifted).transpose(2, 0, 1), x=30, figsize=(8, 8), clipval=(-0.2, 0.2),
                       title='Weighted data')

        # Inversion
        explode_volume((direct_mask * datarec_pdshiftedback).transpose(2, 0, 1), x=30, clipval=(-0.2, 0.2),
                       figsize=(8, 8), title='PD Reconstruction')
        explode_volume((direct_mask * datarec_pdshiftedback).transpose(2, 0, 1) - data.transpose(2, 0, 1), x=30,
                       clipval=(-0.2, 0.2), figsize=(8, 8), title='Error')

        # SNR evolution
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.plot(np.arange(0, niter + 1, jsrnsave), snr_hist_pd[::jsrnsave], 'k', lw=3)
        ax.set_title('SNR')

        plt.show()


if __name__ == "__main__":
    description = '3D POCS Interpolation with off-the-grid receivers'
    main(argparse.ArgumentParser(description=description))
