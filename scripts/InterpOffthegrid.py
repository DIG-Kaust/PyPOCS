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

np.random.seed(5)

def main(parser):

    ######### INPUT PARAMS #########
    parser.add_argument('-c', '--config', type=str, help='Configuration file')
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        setup = yaml.load(stream, Loader=yaml.FullLoader)

    filename = setup['exp']['inputfile']
    plotflag = setup['exp']['plotflag']

    nry = setup['data']['nry']
    nrx = setup['data']['nrx']
    nt = setup['data']['nt']
    dry = setup['data']['dry']
    drx = setup['data']['drx']
    dt = setup['data']['dt']

    perc_subsampling = setup['subsampling']['perc_subsampling']

    vshift = setup['preprocessing']['vshift']
    nwin = setup['preprocessing']['nwin']
    nover = setup['preprocessing']['nover']
    nop = setup['preprocessing']['nop']
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
    print('-------------------------------\n')

    ######### Data loading and preprocessing #########

    # Load data
    nrxorig, nryorig = 177, 90
    data = np.fromfile(filename, dtype='float32')
    data = data.reshape(nt + 75, nrxorig * nryorig)[75:]
    data = data.reshape(nt, nryorig, nrxorig).transpose(2, 1, 0)  # y, x, t
    data = data[:nrx, :nry, :500] / np.max(np.abs(data[:nrx, :nry, :500]))
    ns, nr, nt = data.shape
    t = np.arange(0, nt) * dt

    # Acquisition
    isrc = 3338 * 4  # selected source
    ny, nx, nz = 200, 330, 155
    y, x, z = np.arange(ny) * 15., np.arange(nx) * 15., np.arange(nz) * 15.
    srcx = np.arange(300, x[-1] - 300, 20)
    srcy = np.arange(300, y[-1] - 300, 20)

    SRCY, SRCX = np.meshgrid(srcy, srcx, indexing='ij')
    SRCX, SRCY = SRCX.ravel(), SRCY.ravel()

    recx = np.arange(700, x[-1] - 700, 20)[:nrx]
    recy = np.arange(600, y[-1] - 600, 20)[:nry]

    RECY, RECX = np.meshgrid(recy, recx, indexing='ij')
    RECX, RECY = RECX.ravel(), RECY.ravel()

    # Subsampling locations
    Nsub = int(np.round(ns * perc_subsampling))
    iy = np.arange(ns)
    ix = np.arange(nr)

    iygrid, ixgrid = np.meshgrid(iy, ix, indexing='ij')

    iyava_y = np.random.randint(2, ns - 2, Nsub)
    iyava = np.tile(iyava_y, (nr, 1)).ravel()
    ixava = np.tile(np.arange(nr), (Nsub, 1)).T.ravel()

    mask2d = np.zeros((ns, nr))
    mask2d[iyava, ixava] = 1
    iava2d = np.where(mask2d.ravel() == 1)[0]

    mask3d = np.zeros((ns, nr, nt))
    mask3d[iyava, ixava, :] = 1
    iava3d = np.where(mask3d.ravel() == 1)[0]

    iyava = iygrid.ravel()[iava2d]
    ixava = ixgrid.ravel()[iava2d]
    Nsub = int(len(iava3d) / nt)

    # Create restriction operator
    Rop = Restriction(ns * nr * nt, iava=iava3d.ravel(), dtype='float64')
    dottest(Rop)
    mask = Rop.H * Rop * np.ones_like(data).ravel()
    mask = mask.reshape(ns, nr, nt)

    # Create bilinear operator
    iyava_pert = iyava + np.random.uniform(-0.5, 0.5, Nsub)
    ixava_pert = ixava + np.random.uniform(-0.5, 0.5, Nsub)
    ixava_pert[ixava_pert < 0] = 0.
    ixava_pert[ixava_pert > nr - 2] = nr - 2
    iava3d1 = np.vstack((iyava_pert, ixava_pert))

    Bop = Bilinear(cp_asarray(iava3d1), (ns, nr, nt), dtype='float64')
    dottest(Bop, backend='cupy')

    Bop1 = Bilinear(iava3d1, (ns, nr), dtype='float64')
    BopBopH = (Bop1 @ Bop1.H).todense()

    print(Rop, Bop)

    # Geometry
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.scatter(iygrid, ixgrid, c='k')
    ax.scatter(iyava, ixava, c='r')
    ax.scatter(iava3d1[0], iava3d1[1], c='b')
    ax.set_title('Geometry')
    plt.show()

    # Off-grid data
    datasub = Bop * cp_asarray(data)
    datamasked = Bop.H * datasub

    datasub = cp_asnumpy(datasub)
    datamasked = cp_asnumpy(datamasked)

    # FK spectra
    FFTop = FFTND(dims=[nrx, nry, nt], nffts=[nfft, nfft, nfft], sampling=[drx, dry, dt], real=True)

    datafk = FFTop @ data
    datamaskedfk = FFTop @ datamasked

    # Create direct arrival mask
    direct_mask = np.ones_like(data)
    direct_thresh = 0.02 * np.max(np.abs(data))
    for iy in range(nrx):
        for ix in range(nry):
            direct_mask[iy, ix, :max(0, np.where(np.abs(data[iy, ix]) > direct_thresh)[0][0] - 10)] = 0.

    # Patched fk sparsyfing transform
    dimsd = data.shape
    nop1 = (nop[0], nop[1], nop[2] // 2 + 1)

    nwins, dims, mwins_inends, dwins_inends = patch3d_design(dimsd, nwin, nover, nop1)

    F1op = FFTND(nwin, nffts=nop, real=True, dtype="complex64")
    Srecop = Patch3D(F1op.H, dims, dimsd, nwin, nover, nop1, tapertype='cosine')

    # Shift operator
    shift = np.sqrt((300 - 10) ** 2 +
                    (RECX.reshape(nry, nrx) - SRCX[isrc]) ** 2 +
                    (RECY.reshape(nry, nrx) - SRCY[isrc]) ** 2) / vshift

    Shiftop = Shift((nrx, nry, nt), shift=-shift.T, sampling=dt, axis=2)

    # Shift data
    datashifted = np.real(Shiftop * data)
    datareshifted = np.real(Shiftop.H * datashifted)

    # Shift subsampled data
    datashiftedsub = Bop * cp_asarray(datashifted)
    datashiftedmasked = Bop.H * datashiftedsub

    datashiftedsub = cp_asnumpy(datashiftedsub)
    datashiftedmasked = cp_asnumpy(datashiftedmasked)

    datashiftedfk = FFTop @ datashifted
    datashiftedmaskedfk = FFTop @ datashiftedmasked

    # Find normalization factors
    weightshifted = 1. / (sp.ndimage.gaussian_filter(np.abs(sp.signal.hilbert(datashifted)), sigma=4) + 1e-3)
    weightshifted[:, :200][weightshifted[:, :200] > 300] = 300  # remove infinite values when there is no signal
    weightshifted /= (weightshifted.max() / 20)

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

    ppd = FFTop * xpd.ravel()
    ppd = ppd.reshape(nfft, nfft, nfft // 2 + 1)
    errpd = [np.linalg.norm(datashifted - datashiftedmasked / weightshifted) / np.linalg.norm(
        datashifted)] + callback.err
    snr_hist_pd = 20 * np.log10(1 / np.array(errpd))

    # Renormalize and shift back reconstructed data
    datarec_pdshiftedback = cp_asnumpy(np.real(Shiftop.H * (xpd.reshape(nrx, nry, nt) / cp_asarray(weightshifted))))

    print(metrics(data, direct_mask*datarec_pdshiftedback, itmin=0, verb=True))

    ######### Plotting #########

    if plotflag:
        # Geometry
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.scatter(iygrid, ixgrid, c='k')
        ax.scatter(iyava, ixava, c='r')
        ax.scatter(iava3d1[0], iava3d1[1], c='b')
        ax.set_title('Geometry')

        explode_volume(data.transpose(2, 0, 1), x=96, clipval=(-0.2, 0.2), figsize=(8, 8), title='Full data')
        explode_volume(datamasked.transpose(2, 0, 1), x=96, clipval=(-0.2, 0.2), figsize=(8, 8),
                       title='Subsampled data')

        explode_volume(np.fft.fftshift(np.abs(datafk[..., :55]), axes=(0, 1)).transpose(2, 0, 1), t=25,
                       tlim=[0, FFTop.fs[-1][55]], tlabel='f(Hz)',
                       cmap='jet', figsize=(8, 8), clipval=(0, 0.2))

        explode_volume(np.fft.fftshift(np.abs(datamaskedfk[..., :55]), axes=(0, 1)).transpose(2, 0, 1), t=30,
                       tlim=[0, FFTop.fs[-1][55]], tlabel='f(Hz)',
                       cmap='jet', figsize=(8, 8), clipval=(0, 0.2))

        explode_volume(direct_mask.transpose(2, 0, 1), x=30, cmap='gray_r', clipval=(0, 1), figsize=(8, 8),
                       title='Direct Mask')
        explode_volume((direct_mask * data).transpose(2, 0, 1), x=30, clipval=(-0.2, 0.2), figsize=(8, 8),
                       title='Masked data')

        plt.figure(figsize=(8, 3))
        plt.imshow(shift)
        plt.axis('tight')
        plt.colorbar()
        plt.title('Shift')

        explode_volume(data.transpose(2, 0, 1), t=40, x=95, y=45, clipval=(-0.2, 0.2), figsize=(8, 8), title='Data')
        explode_volume(datashifted.transpose(2, 0, 1), t=40, x=95, y=45, clipval=(-0.2, 0.2), figsize=(8, 8),
                       title='Shifted data')

        explode_volume(np.fft.fftshift(np.abs(datashiftedfk[..., :70]), axes=(0, 1)).transpose(2, 0, 1), t=30,
                       cmap='jet', figsize=(8, 8), clipval=(0, 0.2))

        explode_volume(np.fft.fftshift(np.abs(datashiftedmaskedfk[..., :70]), axes=(0, 1)).transpose(2, 0, 1), t=30,
                       cmap='jet', figsize=(8, 8), clipval=(0, 0.2))

        explode_volume(weightshifted.transpose(2, 0, 1), x=96, figsize=(8, 8), clipval=(0, 10),
                       cmap='jet', title='Data weight')

        explode_volume((datashifted * weightshifted).transpose(2, 0, 1), x=96, figsize=(8, 8), clipval=(-0.2, 0.2),
                       title='Weighted data')

        explode_volume(data.transpose(2, 0, 1), x=30, clipval=(-0.2, 0.2), figsize=(8, 8), title='Data')
        explode_volume((direct_mask * datarec_pdshiftedback).transpose(2, 0, 1), x=30, clipval=(-0.2, 0.2),
                       figsize=(8, 8), title='PD')
        explode_volume((direct_mask * datarec_pdshiftedback).transpose(2, 0, 1) - data.transpose(2, 0, 1), x=30,
                       clipval=(-0.2, 0.2), figsize=(8, 8), title='Error')

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        ax.plot(np.arange(0, niter + 1, jsrnsave), snr_hist_pd[::jsrnsave], 'k', lw=3)
        ax.set_title('SNR')

        plt.show()

if __name__ == "__main__":
    description = '3D POCS Interpolation with off-the-grid receivers'
    main(argparse.ArgumentParser(description=description))
