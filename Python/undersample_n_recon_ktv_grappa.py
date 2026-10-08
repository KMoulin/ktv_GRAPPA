"""
undersample_n_recon_ktv_grappa.py
=================================
Python port of ``Undersample_n_recon_KTV_GRAPPA.m``.

Example (k-t-v GRAPPA, coil-combined image)::

    img = undersample_n_recon_ktv_grappa(kspace, [3, 3], coil_sens=True, mode=2)

Example (k-t-v GRAPPA, k-space only)::

    k_rec = undersample_n_recon_ktv_grappa(kspace, [3, 3], mode=2)
"""
from __future__ import annotations

import time

import numpy as np

from coil_utils import align_cc_mtx, calc_gcc_mtx, cc, fft3c
from grappa_5d_ktv import (GRAPPA5DKtv, free_gpu_memory, get_xp, matlab_round_mid,
                           mslice)


def _coil_combine(k_recon, sens):
    img = np.fft.fftshift(fft3c(k_recon), axes=(1, 2))
    return np.sum(img * np.conj(sens), axis=3)


def undersample_n_recon_ktv_grappa(kspace, listR, coil_sens=None, mode=2, listV=None,
                                   listT=None, undersample=0, nc_cc=None, lam=None,
                                   force_sampling_mask=False, ridge_tikhonov=None,
                                   acs_ext=None, use_gpu=None, verbose=True,
                                   debug_plot=False, max_gb=4.0):
    """GRAPPA / k-t GRAPPA / k-t-v GRAPPA reconstruction of 4D-flow data.

    Parameters
    ----------
    kspace : complex array [x, y, z, coil, venc, time]
    listR : (Ry, Rz) acceleration factors
    coil_sens : None -> return reconstructed k-space;
                True -> estimate coil maps (local eigenvector method) and return
                        coil-combined images;
                array [x, y, z, coil] -> use these maps (replaced by estimated
                        maps if coil compression is used, as in MATLAB).
    mode : 0 GRAPPA, 1 k-t GRAPPA, 2 k-t-v GRAPPA (default)
    listV, listT : 0-based venc / cardiac-phase indices to reconstruct (default all)
    undersample : 0 no; 1 or 2 retrospectively undersample ``kspace`` at listR
                  (interleaving mode of ``undersample_kspace``), ACS = 21x17 lines
    nc_cc : number of virtual coils for geometric coil compression (None = off)
    lam : Tikhonov weight relative to the mean diagonal of S S^H. None keeps the
          value hard-coded in the MATLAB recon (1e-1); 0 disables regularisation.
          (In MATLAB this argument was only forwarded to the CUDA mex.)
    force_sampling_mask : impose the regular lattice mask instead of the
                          detected one (zeroing the ACS block)
    ridge_tikhonov : unused (kept for signature compatibility)
    acs_ext : optional separately acquired ACS block (Siemens style), pasted in
              the centre with ``add_acs_siemens`` (dead code in the MATLAB version)
    use_gpu : None auto-detect CuPy, True force GPU, False force CPU
    max_gb : memory budget per interpolation block (GB), as in MATLAB (4 GB)

    Returns
    -------
    [x, y, z, venc, time] complex64 images if coil_sens is given,
    otherwise [x, y, z, coil, venc, time] complex64 k-space.
    """
    G = GRAPPA5DKtv
    xp = get_xp(use_gpu)

    kspace = np.array(kspace, copy=True)
    if not np.iscomplexobj(kspace):
        kspace = kspace.astype(np.complex64)
    kspace[np.isinf(kspace)] = 0
    if kspace.ndim < 6:
        kspace = kspace.reshape(kspace.shape + (1,) * (6 - kspace.ndim))
    Nx, Ny, Nz, n_coil, Vmax, Tmax = kspace.shape

    # ---- coil-combination options
    if coil_sens is None or (np.ndim(coil_sens) == 0 and not coil_sens):
        coil_combine, estimate_sens = False, False
    elif np.ndim(coil_sens) == 0:
        coil_combine, estimate_sens = True, True
    else:
        coil_combine, estimate_sens = True, False
        coil_sens = np.asarray(coil_sens)

    listV = range(Vmax) if listV is None else listV
    listT = range(Tmax) if listT is None else listT
    nc_cc = n_coil if nc_cc is None else int(nc_cc)
    lam_rel = 1e-1 if lam is None else lam
    Ry, Rz = int(listR[0]), int(listR[1])

    # ---- retrospective undersampling
    if undersample:
        kspace, _ = G.undersample_kspace(kspace, Ry, Rz, (10, 8), undersample)

    # ---- detect the ACS block (points sampled in every venc/time frame)
    mask_0 = (np.abs(kspace.mean(axis=3, keepdims=True)) != 0).astype(np.float32)
    mask = mask_0[matlab_round_mid(Nx), :, :, 0, :, :].min(axis=(2, 3))     # [Ny, Nz]
    idy, idz = np.nonzero(mask)
    if idy.size == 0:
        raise ValueError("No fully sampled (ACS) region found in kspace.")
    acs_size = [(idy.max() - idy.min()) / 2, (idz.max() - idz.min()) / 2]

    if acs_ext is not None:
        kspace, acs_size = G.add_acs_siemens(kspace, acs_ext, Ry, Rz)

    # ---- geometric coil compression
    cc_applied = nc_cc != n_coil
    if cc_applied:
        calib = kspace[:, mslice(Ny / 2 - acs_size[0], Ny / 2 + acs_size[0]),
                       mslice(Nz / 2 - acs_size[1], Nz / 2 + acs_size[1]), :, 0, 0]
        gccmtx = calc_gcc_mtx(calib, dim=0, ws=5)
        gccmtx_aligned = align_cc_mtx(gccmtx[:, :nc_cc, :])
        kspace_cc = np.zeros((Nx, Ny, Nz, nc_cc, Vmax, Tmax), dtype=kspace.dtype)
        for v in range(Vmax):
            for t in range(Tmax):
                kspace_cc[..., v, t] = cc(kspace[..., v, t], gccmtx_aligned, dim=0)
        kspace = kspace_cc
        n_coil = nc_cc
        del kspace_cc, calib, gccmtx, gccmtx_aligned

    # ---- coil sensitivities (from frame 0,0 of the undersampled k-space, as in MATLAB)
    if coil_combine and (estimate_sens or cc_applied):
        if verbose:
            print("Estimating coil sensitivity maps ...")
        img_xby2 = np.fft.fftshift(fft3c(kspace[..., 0, 0]), axes=(1, 2))
        coil_sens = G.espirit_km(img_xby2.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
        del img_xby2
    if coil_combine and coil_sens.shape != (Nx, Ny, Nz, n_coil):
        raise ValueError(f"coil_sens has shape {coil_sens.shape}, "
                         f"expected {(Nx, Ny, Nz, n_coil)}")

    # ---- ACS, kernel networks, sampling masks
    ACS = kspace[:, idy.min():idy.max() + 1, idz.min():idz.max() + 1]
    net_ktv = G.get_net_ktv(Ry, Rz, n_coil, mode)
    mask_YZ2, list_YZ = G.mask_discovery(kspace, Ry, Rz)
    if force_sampling_mask:
        mask_YZ = mask_YZ2.copy()
        mask_YZ[:, idy.min():idy.max() + 1, idz.min():idz.max() + 1] = 0
    else:
        mask_YZ = mask_YZ2 * (1 - mask_0[:, :, :, 0, :, :])

    # ---- output
    if coil_combine:
        out = np.zeros((Nx, Ny, Nz, Vmax, Tmax), dtype=np.complex64)
    else:
        out = np.zeros((Nx, Ny, Nz, n_coil, Vmax, Tmax), dtype=np.complex64)

    for v in listV:
        for t in listT:
            if Ry == 1 and Rz == 1:
                k_recon = kspace[..., v, t].astype(np.complex64)
            else:
                if verbose:
                    print(f" Velocity {v + 1}/{Vmax} Cardiac phase {t + 1}/{Tmax}"
                          f"  [Ry={Ry};Rz={Rz}]")
                k_comp = G.create_composite(kspace, list_YZ, Ry, Rz, v, t, xp=xp,
                                            verbose=verbose)
                acs_comp = G.create_composite(ACS, list_YZ, Ry, Rz, v, t, xp=xp)
                k_in = xp.asarray(kspace[..., v, t], dtype=xp.complex64)
                acs_in = xp.asarray(ACS[..., v, t], dtype=xp.complex64)

                t0 = time.perf_counter()
                k_recon = G.recon_n_train_5d(k_comp, k_in, acs_comp, acs_in,
                                             mask_YZ[..., v, t], net_ktv, Ry, Rz,
                                             lam_rel=lam_rel, max_gb=max_gb,
                                             verbose=verbose, debug_plot=debug_plot)
                if verbose:
                    print(f"  time to recon one 3D volume: {time.perf_counter() - t0:.2f} s")
                del k_comp, acs_comp, k_in, acs_in
                if xp is not np:
                    free_gpu_memory()

            if coil_combine:
                out[..., v, t] = _coil_combine(k_recon, coil_sens)
            else:
                out[..., v, t] = k_recon
    return out
