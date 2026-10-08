"""
coil_utils.py
=============
FFT helpers and coil-compression routines used by the k-t-v GRAPPA code.

They replace external MATLAB dependencies that the original code calls:

* ``fftc`` / ``ifftc`` / ``sos``          -> ESPIRiT toolbox (M. Lustig) utilities
* ``calcGCCMtx`` / ``alignCCMtx`` / ``CC`` -> geometric coil compression
                                              (Zhang et al., MRM 2013), ESPIRiT v0.3
* ``fft3c``                               -> your own helper. ASSUMED here to be a
                                              centred, orthonormal 3-D FFT over the
                                              first three axes. If your MATLAB fft3c is
                                              defined differently, change it here.

All ``axis`` / ``dim`` arguments are 0-based (MATLAB dim 1 -> Python 0).
"""
import numpy as np


# --------------------------------------------------------------------------- FFTs
def fftc(x, axis):
    """Centred, orthonormal 1-D FFT along ``axis``."""
    x = np.fft.ifftshift(x, axes=axis)
    x = np.fft.fft(x, axis=axis, norm="ortho")
    return np.fft.fftshift(x, axes=axis)


def ifftc(x, axis):
    """Centred, orthonormal 1-D inverse FFT along ``axis``."""
    x = np.fft.ifftshift(x, axes=axis)
    x = np.fft.ifft(x, axis=axis, norm="ortho")
    return np.fft.fftshift(x, axes=axis)


def fft3c(x, axes=(0, 1, 2)):
    """Centred, orthonormal 3-D FFT over ``axes`` (assumed definition of fft3c)."""
    x = np.fft.ifftshift(x, axes=axes)
    x = np.fft.fftn(x, axes=axes, norm="ortho")
    return np.fft.fftshift(x, axes=axes)


def ifft3c(x, axes=(0, 1, 2)):
    """Centred, orthonormal 3-D inverse FFT over ``axes``."""
    x = np.fft.ifftshift(x, axes=axes)
    x = np.fft.ifftn(x, axes=axes, norm="ortho")
    return np.fft.fftshift(x, axes=axes)


def sos(x, axis):
    """Root sum of squares along ``axis``."""
    return np.sqrt(np.sum(np.abs(x) ** 2, axis=axis))


# ------------------------------------------------------- geometric coil compression
def calc_gcc_mtx(calib, dim=0, ws=1):
    """Port of ``calcGCCMtx``.

    calib : [kx, ky, (kz), coil] calibration data
    dim   : 0-based axis along which compression matrices vary (MATLAB dim-1)
    ws    : odd sliding-window size
    Returns mtx [Nc, min(Nc, ws*Ny*Nz), N_dim]
    """
    calib = np.asarray(calib)
    if calib.ndim == 3:
        if dim == 2:
            raise ValueError("Cannot compress along the 3rd dimension of 2D data")
        calib = calib[:, :, None, :]
    elif calib.ndim != 4:
        raise ValueError("calib must be [kx, ky, (kz), coil]")

    calib = np.moveaxis(calib, dim, 0)
    calib = ifftc(calib, axis=0)                      # hybrid space
    Nx, Ny, Nz, Nc = calib.shape

    mtx = np.zeros((Nc, min(Nc, ws * Ny * Nz), Nx), dtype=np.complex128)

    # centred zero padding along x (same as ESPIRiT zpad)
    total = Nx + ws - 1
    before = total // 2 - Nx // 2
    zp = np.zeros((total, Ny, Nz, Nc), dtype=np.complex128)
    zp[before:before + Nx] = calib

    for n in range(Nx):
        blk = zp[n:n + ws].reshape(-1, Nc)
        _, _, vh = np.linalg.svd(blk, full_matrices=False)
        mtx[:, :, n] = vh.conj().T
    return mtx


def align_cc_mtx(mtx, ncc=None):
    """Port of ``alignCCMtx``: align compression matrices along the varying axis."""
    mtx = np.array(mtx, copy=True)
    _, sy, nc = mtx.shape
    if ncc is None:
        ncc = sy
    n0 = max(nc // 2 - 1, 0)          # MATLAB floor(nc/2), 1-based -> 0-based

    def _align(order):
        A0 = mtx[:, :ncc, n0]
        for n in order:
            A1 = mtx[:, :ncc, n]
            C = A1.conj().T @ A0
            U, _, Vh = np.linalg.svd(C, full_matrices=False)
            P = Vh.conj().T @ U.conj().T              # V*U'
            mtx[:, :ncc, n] = A1 @ P.conj().T
            A0 = mtx[:, :ncc, n]

    _align(range(n0 - 1, -1, -1))      # backwards to first slice
    _align(range(n0 + 1, nc))          # forwards to last slice
    return mtx


def cc(data, mtx, dim=0):
    """Port of ``CC``: apply compression matrices. data [kx, ky, (kz), coil]."""
    data = np.asarray(data)
    is2d = data.ndim == 3
    if is2d:
        data = data[:, :, None, :]
    data = np.moveaxis(data, dim, 0)
    d = ifftc(data, axis=0)
    res = np.einsum("xyzc,ckx->xyzk", d, mtx, optimize=True)
    res = fftc(res, axis=0)
    res = np.moveaxis(res, 0, dim)
    return res[:, :, 0, :] if is2d else res
