"""
grappa_5d_ktv.py
================
Python port of the MATLAB class ``GRAPPA_5D_ktv`` (k-t-v GRAPPA for 4D flow).

Conventions (differences from MATLAB)
-------------------------------------
* Axis order is unchanged: k-space is [x, y, z, coil, venc, time].
* Every index is 0-based:
    - sampling shifts returned by ``mask_discovery`` / ``find_shift`` are the
      MATLAB values minus 1;
    - the coil column (and the venc/time index columns of ``get_net_ktv2``)
      of a kernel network is 0-based.
* Kernel networks (NetKTV) and saved weights have ``Ry*Rz`` pages. Page 0 (the
  "already acquired" kernel) is unused, so page ``cK`` here is exactly
  MATLAB's ``NetKTV(:,:,cK)``.
* GPU: if CuPy is installed and a GPU is visible, the heavy steps (patch
  gathering, weight fitting, interpolation) run on the GPU; otherwise NumPy.
  Pass CuPy arrays to the recon functions to run on the GPU.

Not ported: ``Mask_discovery_old`` (incomplete, its result was never used) and
``Undersample_comp_mVENC_old`` (superseded by ``undersample_comp_mvenc``).
"""
from __future__ import annotations

import math
import time
import warnings

import numpy as np
from scipy.ndimage import uniform_filter

from coil_utils import sos

try:
    import cupy as cp
except Exception:  # pragma: no cover  (ImportError or a broken CUDA install)
    cp = None

_GPU_STATUS = None   # cached (ok: bool, reason: str)


# =============================================================== backend helpers
def _gpu_self_test():
    """Run the same kinds of GPU operations the recon uses (strided assignment,
    fancy indexing, matmul, solve). Counting devices is not enough: a CuPy build
    that does not match the CUDA driver/toolkit only fails at the first kernel."""
    global _GPU_STATUS
    if _GPU_STATUS is not None:
        return _GPU_STATUS
    if cp is None:
        _GPU_STATUS = (False, "CuPy is not installed")
        return _GPU_STATUS
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            raise RuntimeError("no CUDA device found")
        a = cp.zeros((4, 6, 6, 2), dtype=cp.complex64)
        a[:, 1::2, ::3, :] = cp.asarray(np.ones((4, 3, 2, 2), np.complex64))
        idx = cp.asarray([0, 1, 2])
        b = a[idx, idx, idx, :].T
        m = b @ b.conj().T + cp.eye(2, dtype=cp.complex64)
        cp.linalg.solve(m, m)
        cp.cuda.Stream.null.synchronize()
        _GPU_STATUS = (True, "")
    except Exception as e:
        _GPU_STATUS = (False, f"{type(e).__name__}: {e}")
    return _GPU_STATUS


def gpu_available():
    return _gpu_self_test()[0]


def get_xp(use_gpu=None):
    """Return cupy or numpy.

    use_gpu=None : use the GPU if CuPy works, otherwise warn and use NumPy
    use_gpu=True : require the GPU (raises with the CuPy error if it fails)
    use_gpu=False: NumPy only
    """
    if use_gpu is False:
        return np
    ok, reason = _gpu_self_test()
    if ok:
        return cp
    if use_gpu:
        raise RuntimeError(f"use_gpu=True but the GPU is not usable: {reason}")
    if cp is not None:
        warnings.warn(f"CuPy is installed but not usable ({reason}); "
                      "falling back to NumPy on the CPU.")
    return np


def array_module(a):
    return cp.get_array_module(a) if cp is not None else np


def to_numpy(a):
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def free_gpu_memory():
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()


def _sync(xp):
    if cp is not None and xp is cp:
        cp.cuda.Stream.null.synchronize()


# ============================================================= indexing helpers
def mslice(lo, hi):
    """MATLAB ``lo:hi`` (1-based, inclusive) -> Python slice."""
    return slice(max(int(math.ceil(lo)) - 1, 0), int(math.floor(hi)))


def matlab_mid(n):
    """0-based index of MATLAB ``end/2`` (the original assumes even sizes)."""
    return max(n // 2 - 1, 0)


def matlab_round_mid(n):
    """0-based index of MATLAB ``round(end/2)``."""
    return max(int(math.floor(n / 2 + 0.5)) - 1, 0)


def _first_page(a, nd):
    """a[:, ..., :, 0, 0, ...] keeping the first ``nd`` axes (MATLAB linear-index style)."""
    if a.ndim > nd:
        return a[(slice(None),) * nd + (0,) * (a.ndim - nd)]
    return a


def mrdivide(A, B):
    """MATLAB ``A / B`` (= A @ inv(B))."""
    xp = array_module(A)
    try:
        return xp.linalg.solve(B.T, A.T).T
    except np.linalg.LinAlgError:
        return xp.linalg.lstsq(B.T, A.T, rcond=None)[0].T


# ====================================================== local eigen coil maps
def _local_eig_maps(data, w=5, max_bytes=512 * 2**20):
    """Dominant eigenvector / sqrt(eigenvalue) of the local coil covariance.

    Vectorised equivalent of the per-voxel ``eigs`` loop in ESPIRIT_KM: the
    (2w+1)^d window truncated at the borders equals a zero-padded box sum.
    data: [..spatial.., coil]. Returns S (same shape) and M (spatial shape).
    """
    data = np.asarray(data, dtype=np.complex128)
    sp, Nc = data.shape[:-1], data.shape[-1]
    size = 2 * w + 1
    vol = float(size ** len(sp))
    S = np.zeros(data.shape, dtype=np.complex128)
    M = np.zeros(sp)

    per_row = int(np.prod(sp[1:])) * Nc * Nc * 16 * 3
    slab = max(1, max_bytes // max(per_row, 1))
    for x0 in range(0, sp[0], slab):
        x1 = min(sp[0], x0 + slab)
        h0, h1 = max(0, x0 - w), min(sp[0], x1 + w)      # slab plus halo
        d = data[h0:h1]
        R = np.empty(d.shape[:-1] + (Nc, Nc), dtype=np.complex128)
        for a in range(Nc):
            for b in range(a, Nc):
                p = d[..., a] * np.conj(d[..., b])     # = conj(K'K)(a,b) contribution
                s = (uniform_filter(p.real, size, mode="constant")
                     + 1j * uniform_filter(p.imag, size, mode="constant")) * vol
                R[..., a, b] = s
                if a != b:
                    R[..., b, a] = np.conj(s)
        R = R[x0 - h0:x0 - h0 + (x1 - x0)]
        evals, evecs = np.linalg.eigh(R)
        v = evecs[..., :, -1]                          # largest eigenvalue
        S[x0:x1] = v * np.exp(-1j * np.angle(v[..., :1]))
        M[x0:x1] = np.sqrt(np.clip(evals[..., -1], 0, None))
    return S, M


# ================================================================ main class
class GRAPPA5DKtv:
    """Port of ``GRAPPA_5D_ktv`` (static methods)."""

    NetX = (1, 0, -1)
    NetT = (0, 1, 2)
    NetVenc = (0, 1, 2, 3)
    NetT0 = (0,)
    NetVenc0 = (0,)
    MaxVencPts = 4
    MaxTimePts = 3

    # The MATLAB getListofExample uses ``[Nx,Ny,Nz]=size(ACS)`` on a 4-D array,
    # so Nz silently becomes Nz*Ncoil and the upper z-boundary check is too
    # loose (training patches then contain zero-filled neighbours). False = fixed
    # behaviour; set True to reproduce MATLAB results exactly.
    MATLAB_COMPAT_BOUNDS = False

    # ------------------------------------------------------------ recon cores
    @classmethod
    def recon_n_train_5d(cls, k_composite, k_R, acs_composite, acs, mask_R, net_ktv,
                         Ry, Rz, lam_rel=1e-1, max_gb=4.0, verbose=False, debug_plot=False):
        """Port of ``Recon_n_Train_5D``: train one kernel per missing-point type
        on the ACS and fill the missing points of a single (venc, time) frame.

        k_composite, acs_composite : [Nx,Ny,Nz,Nc] composite k-space / ACS
        k_R, acs                   : [Nx,Ny,Nz,Nc] current frame k-space / ACS
                                     (acs=None -> acs_composite holds external
                                     weights [Nc, K, Ry*Rz], page cK per kernel)
        mask_R                     : [Nx,Ny,Nz] kernel index of every point
        lam_rel                    : Tikhonov weight relative to mean diag(S S^H)
                                     (MATLAB hard-codes 1e-1 here); 0 = none
        """
        xp = array_module(k_composite)
        k_recon = xp.array(_first_page(k_R, 4), dtype=xp.complex64, copy=True)
        external_w = acs is None
        n_k = Ry * Rz
        for cK in range(1, n_k):
            net = net_ktv[:, :, cK]
            if external_w:
                W = xp.asarray(acs_composite[:, :, cK], dtype=xp.complex64)
            else:
                W = cls._train_weights(acs_composite, acs, net, lam_rel, debug_plot, cK)
            cls._apply_kernel(k_recon, k_composite, mask_R, cK, net, W, max_gb)
            if verbose:
                print(f"    kernel {cK}/{n_k - 1}", end="\r", flush=True)
        if verbose:
            print()
        return to_numpy(k_recon)

    @classmethod
    def recon_n_time_5d(cls, k_composite, k_R, acs_composite, acs, mask_R, net_ktv,
                        Ry, Rz, lam_rel=1e-3, max_gb=4.0):
        """Port of ``Recon_n_Time_5D``: same as recon_n_train_5d, with timings.

        Returns (k_recon, time_save); time_save[cK] = [patch Sn, patch Xn,
        weights, patch Sr, apply]. Row 0 unused.
        """
        xp = array_module(k_composite)
        k_recon = xp.array(_first_page(k_R, 4), dtype=xp.complex64, copy=True)
        n_k = Ry * Rz
        time_save = np.zeros((n_k, 5))
        for cK in range(1, n_k):
            net = net_ktv[:, :, cK]
            lst = cls.get_list_of_example(acs, net)
            t0 = time.perf_counter()
            Sn = cls.patch(acs_composite, lst, net)
            _sync(xp)
            time_save[cK, 0] = time.perf_counter() - t0
            t0 = time.perf_counter()
            Xn = cls.patch_acs(acs, lst)
            _sync(xp)
            time_save[cK, 1] = time.perf_counter() - t0
            t0 = time.perf_counter()
            W = cls._solve_weights(Sn, Xn, lam_rel)
            _sync(xp)
            time_save[cK, 2] = time.perf_counter() - t0
            del Sn, Xn
            time_save[cK, 3:5] = cls._apply_kernel(k_recon, k_composite, mask_R, cK, net, W, max_gb)
        return to_numpy(k_recon), time_save

    @classmethod
    def recon_n_weight_5d(cls, k_composite, k_R, acs_composite, acs, mask_R, net_ktv,
                          Ry, Rz, lam_rel=1e-3):
        """Port of ``Recon_n_Weight_5D``: only trains and returns the weights.

        Returns (k_R unchanged, w_save [Nc, K, Ry*Rz]); page 0 unused. w_save can
        be fed back to recon_n_train_5d as ``acs_composite`` with ``acs=None``.
        """
        xp = array_module(acs_composite)
        k_recon = to_numpy(_first_page(k_R, 4)).astype(np.complex64)
        Nc = acs.shape[3]
        n_k = Ry * Rz
        w_save = np.zeros((Nc, net_ktv.shape[0], n_k), dtype=np.complex64)
        for cK in range(1, n_k):
            W = cls._train_weights(acs_composite, acs, net_ktv[:, :, cK], lam_rel, False, cK)
            w_save[:, :, cK] = to_numpy(W)
        if xp is not np:
            free_gpu_memory()
        return k_recon, w_save

    # ------------------------------------------------------------ recon helpers
    @staticmethod
    def _solve_weights(Sn, Xn, lam_rel):
        xp = array_module(Sn)
        SSSh = Sn @ Sn.conj().T
        A = Xn @ Sn.conj().T
        if lam_rel:
            lam = lam_rel * float(xp.real(xp.trace(SSSh))) / SSSh.shape[0]
            SSSh = SSSh + lam * xp.eye(SSSh.shape[0], dtype=SSSh.dtype)
        return mrdivide(A, SSSh)

    @classmethod
    def _train_weights(cls, acs_composite, acs, net, lam_rel, debug_plot=False, kernel=None):
        lst = cls.get_list_of_example(acs, net)
        if lst.shape[0] == 0:
            raise ValueError(f"No ACS training example for kernel {kernel}: the ACS "
                             "region is too small for the kernel footprint.")
        Sn = cls.patch(acs_composite, lst, net)      # [(Nc*patch), examples]
        Xn = cls.patch_acs(acs, lst)                 # [Nc, examples]
        if debug_plot:
            cls._plot_singular_values(Sn, kernel)
        return cls._solve_weights(Sn, Xn, lam_rel)

    @staticmethod
    def _plot_singular_values(Sn, kernel):
        import matplotlib.pyplot as plt
        SSSh = to_numpy(Sn @ Sn.conj().T)
        s = np.sort(np.linalg.svd(SSSh, compute_uv=False))[::-1]
        plt.figure()
        plt.semilogy(s, "o-")
        plt.xlabel("index")
        plt.ylabel("singular value (log scale)")
        plt.title(f"Singular value spectrum of S*S^H (kernel {kernel})")
        plt.grid(True)
        plt.show(block=False)

    @classmethod
    def _apply_kernel(cls, k_recon, k_composite, mask_R, cK, net, W, max_gb=4.0):
        """Fill every point with mask_R == cK. Returns (t_patch, t_apply)."""
        xp = array_module(k_recon)
        Nc = k_recon.shape[3]
        pts = np.argwhere(to_numpy(mask_R) == cK)     # [P, 3]
        ly = pts.shape[0]
        if ly == 0:
            return 0.0, 0.0
        # Same memory heuristic as MATLAB (lx = length(NetKTV)*Nc, 16 B/elem),
        # but with ceil() so that no point is dropped by the segmentation.
        gb = net.shape[0] * Nc * ly * 16 / 1024**3
        nseg = math.ceil(gb / max_gb) if gb > max_gb else 1
        block = math.ceil(ly / nseg)
        t_patch = t_apply = 0.0
        for s in range(0, ly, block):
            sub = xp.asarray(pts[s:s + block])
            t0 = time.perf_counter()
            Sr = cls.patch(k_composite, sub, net)
            _sync(xp)
            t1 = time.perf_counter()
            Xr = W @ Sr                                # [Nc, points]
            k_recon[sub[:, 0], sub[:, 1], sub[:, 2], :] = Xr.T
            _sync(xp)
            t_patch += t1 - t0
            t_apply += time.perf_counter() - t1
            del Sr, Xr
        return t_patch, t_apply

    # ------------------------------------------------------------ patch gathers
    @staticmethod
    def patch(vect_mat, list_points, net, max_elems=2**23):
        """Port of ``Patch_GPU``: gather the kernel neighbourhood of every point.

        vect_mat [Nx,Ny,Nz,Nc], list_points [P,3], net [K,>=4] (dx,dy,dz,coil).
        Returns [K, P]; neighbours outside the volume are 0.
        """
        xp = array_module(vect_mat)
        v = xp.ascontiguousarray(_first_page(vect_mat, 4))
        Nx, Ny, Nz, Nc = v.shape
        flat = v.reshape(-1)
        pts = xp.asarray(list_points, dtype=xp.int64)
        net = xp.asarray(net, dtype=xp.int64)
        K, P = net.shape[0], pts.shape[0]
        out = xp.zeros((K, P), dtype=v.dtype)
        dx, dy, dz, dc = (net[:, i:i + 1] for i in range(4))
        step = max(1, max_elems // max(K, 1))
        for s in range(0, P, step):
            p = pts[s:s + step]
            x = dx + p[None, :, 0]
            y = dy + p[None, :, 1]
            z = dz + p[None, :, 2]
            valid = (x >= 0) & (x < Nx) & (y >= 0) & (y < Ny) & (z >= 0) & (z < Nz)
            lin = ((x * Ny + y) * Nz + z) * Nc + dc
            lin = xp.where(valid, lin, 0)
            out[:, s:s + step] = xp.where(valid, flat[lin], 0)
        return out

    @staticmethod
    def patch_acs(vect_mat, list_points):
        """Port of ``Patch_GPU_ACS``: all coils at each point -> [Nc, P]."""
        xp = array_module(vect_mat)
        v = _first_page(vect_mat, 4)
        p = xp.asarray(list_points, dtype=xp.int64)
        return v[p[:, 0], p[:, 1], p[:, 2], :].T

    @classmethod
    def get_list_of_example(cls, acs, net):
        """Port of ``getListofExample``: ACS points (non-zero in coil 0) whose
        whole kernel neighbourhood lies inside the ACS. Returns [P,3] (x,y,z)."""
        xp = array_module(acs)
        Nx, Ny, Nz = acs.shape[:3]
        Nz_bound = Nz * int(np.prod(acs.shape[3:])) if cls.MATLAB_COMPAT_BOUNDS else Nz
        net = to_numpy(net)
        mn, mx = net[:, :3].min(0), net[:, :3].max(0)
        lo = [max(0, -int(mn[i])) for i in range(3)]
        hi = [min(n, n - int(mx[i])) for i, n in enumerate((Nx, Ny, Nz_bound))]
        hi[2] = min(hi[2], Nz)
        if any(h <= l for l, h in zip(lo, hi)):
            return xp.zeros((0, 3), dtype=xp.int64)
        sub = _first_page(acs, 3)[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        ix, iy, iz = xp.nonzero(sub != 0)       # same x-y-z order as MATLAB loops
        return xp.stack([ix + lo[0], iy + lo[1], iz + lo[2]], axis=1).astype(xp.int64)

    # ------------------------------------------------------------ composites
    @classmethod
    def _select_frames(cls, list_mask, Ry, Rz, current_v, current_t):
        """Frame-selection logic of ``Create_Composite_GPU``, reproduced exactly
        (computed in 1-based arithmetic, returned 0-based)."""
        Nv, Nt = list_mask.shape[:2]
        Mv, Mt = min(cls.MaxVencPts, Ry), min(cls.MaxTimePts, Rz)
        err = ("Create_Composite: a composite for R={1} needs {1} different "
               "sampling shifts across the {0} frames, but the detected shifts are "
               f"ky (per venc, time 0) = {list_mask[:, 0, 0].tolist()}, "
               f"kz (per time, venc 0) = {list_mask[0, :, 1].tolist()}. "
               "This method needs sampling interleaved across venc (ky) and time "
               "(kz); the MATLAB version loops forever in this case. Check the "
               "data / the `undersample` option, or use Ry=1 / Rz=1 for that axis.")

        cv, ct = current_v + 1, current_t + 1
        list_venc, n, c, guard = [cv], 1, cv, 0
        while n < Mv:
            c += 1
            if c > Nv:
                c = 1
            if list_mask[c - 1, 0, 0] not in [list_mask[i - 1, 0, 0] for i in list_venc]:
                n += 1
                list_venc.append(c)
            guard += 1
            if guard > 2 * Nv + 2:
                raise RuntimeError(err.format("velocity", Ry))

        list_car, n, guard = [ct], 1, 0
        if ct != 1:
            c = ct + 2
        # (if ct == 1, MATLAB continues from the value left by the venc loop)
        while n < Mt:
            c -= 1
            if c < 1:
                c = Nt - c
            if ct == 1 and c == Nt - 2:   # don't go too far backward for phase 1
                c = ct + 2
            if c > Nt:
                c = 1
            if list_mask[0, c - 1, 1] not in [list_mask[0, i - 1, 1] for i in list_car]:
                n += 1
                list_car.append(c)
            guard += 1
            if guard > 4 * Nt + 4:
                raise RuntimeError(err.format("time", Rz))
        return [i - 1 for i in list_venc], [i - 1 for i in list_car]

    @classmethod
    def create_composite(cls, kspace, list_mask, Ry, Rz, current_v, current_t,
                         no_shift=False, xp=np, verbose=False):
        """Port of ``Create_Composite_GPU``.

        kspace [Nx,Ny,Nz,Nc,Nv,Nt] (numpy), list_mask [Nv,Nt,2] 0-based shifts.
        Returns composite [Nx,Ny,Nz,Nc] complex64 on backend ``xp``.
        """
        kspace = to_numpy(kspace)
        list_mask = to_numpy(list_mask)
        Nx, Ny, Nz, Nc = kspace.shape[:4]
        comp = np.zeros((Nx, Ny, Nz, Nc), dtype=np.complex64)   # built on the host
        lv, lt = cls._select_frames(list_mask, Ry, Rz, current_v, current_t)
        if verbose:
            print(f"  composite using venc frames {lv} and time frames {lt} (0-based)")
        for v in lv:
            for t in lt:
                sy, sz = int(list_mask[v, t, 0]), int(list_mask[v, t, 1])
                if no_shift:
                    # (MATLAB used the y-shift for the z end index here: fixed)
                    src = kspace[:, 0:Ny - sy:Ry, 0:Nz - sz:Rz, :, v, t]
                else:
                    src = kspace[:, sy::Ry, sz::Rz, :, v, t]
                comp[:, sy::Ry, sz::Rz, :] = src
        return comp if xp is np else xp.asarray(comp)          # one transfer

    @classmethod
    def create_composite_t(cls, kspace, list_mask, Ry, Rz, current_v, current_t,
                           no_shift=False, xp=np, verbose=False):
        """Port of ``Create_Composite_GPU_T``: average frames sharing a sampling
        shift, then build the composite. Returns (composite, averaged frame)."""
        Mv_u = int(list_mask[:, :, 0].max()) + 1
        Mt_u = int(list_mask[:, :, 1].max()) + 1
        Nx, Ny, Nz, Nc = kspace.shape[:4]
        k_ave = np.zeros((Nx, Ny, Nz, Nc, Mv_u, Mt_u), dtype=np.complex64)
        for a in range(Mv_u):
            for b in range(Mt_u):
                iv, it = np.nonzero((list_mask[:, :, 0] == a) & (list_mask[:, :, 1] == b))
                # NB: like MATLAB, this averages the Cartesian product iv x it,
                # not only the (iv, it) pairs.
                k_ave[..., a, b] = kspace[..., iv[:, None], it[None, :]].mean(axis=(4, 5))
        list_mask2 = list_mask[:Mv_u, :Mt_u, :]
        a0, b0 = int(list_mask[current_v, current_t, 0]), int(list_mask[current_v, current_t, 1])
        comp = cls.create_composite(k_ave, list_mask2, Ry, Rz, a0, b0, no_shift, xp, verbose)
        return comp, xp.asarray(k_ave[..., a0, b0])

    @staticmethod
    def create_kcomposite(kspace, Ry, Rz, shift_calc, xp=np):
        """Port of ``Create_KComposite_GPU``. ``shift_calc`` [Nv,Nt,2] (0-based)
        is required: the MATLAB fallback referenced undefined variables."""
        if shift_calc is None:
            raise ValueError("shift_calc is required (see find_shift)")
        comp = xp.zeros(kspace.shape[:4], dtype=xp.complex64)
        for v in range(min(kspace.shape[4], Ry)):
            for t in range(min(kspace.shape[5], Rz)):
                sy, sz = int(shift_calc[v, t, 0]), int(shift_calc[v, t, 1])
                comp[:, sy::Ry, sz::Rz, :] = xp.asarray(kspace[:, sy::Ry, sz::Rz, :, v, t])
        return comp

    # ------------------------------------------------------------ masks/shifts
    @staticmethod
    def mask_discovery(k_compose, Ry, Rz):
        """Port of ``Mask_discovery``.

        Returns mask_YZ [Nx,Ny,Nz,Nv,Nt] (kernel index cK of every point, 0 =
        acquired lattice) and listKvc [Nv,Nt,2] (0-based ky/kz sampling shift).
        """
        Nx, Ny, Nz, Nc, Nv, Nt = k_compose.shape
        k_mask = np.any(k_compose != 0, axis=(0, 3))          # [Ny,Nz,Nv,Nt]
        list_kvc = np.zeros((Nv, Nt, 2), dtype=np.int64)
        for v in range(Nv):
            for t in range(Nt):
                best = 0
                for sy in range(Ry):
                    for sz in range(Rz):
                        cnt = int(k_mask[sy::Ry, sz::Rz, v, t].sum())
                        if cnt > best:
                            best = cnt
                            list_kvc[v, t] = (sy, sz)

        mask_YZ = np.zeros((Nx, Ny, Nz, Nv, Nt), dtype=np.int32)
        cK = 0
        for cZ in range(Rz):
            for cY in range(Ry):
                for v in range(Nv):
                    for t in range(Nt):
                        ys = np.arange((list_kvc[v, t, 0] + cY) % Ry, Ny, Ry)
                        zs = np.arange((list_kvc[v, t, 1] + cZ) % Rz, Nz, Rz)
                        frame = mask_YZ[:, :, :, v, t]
                        frame[:, ys[:, None], zs[None, :]] = cK
                cK += 1
        return mask_YZ, list_kvc

    @staticmethod
    def find_shift(kspace, Ry, Rz):
        """Port of ``find_shift``: best-matching (0-based) lattice shift per frame."""
        Nx, Ny, Nz, Nc, Nv, Nt = kspace.shape[:6]
        km = (sos(_first_page(kspace, 6)[matlab_mid(Nx)], axis=2) != 0).astype(float)
        shift = np.zeros((Nv, Nt, 2), dtype=np.int64)
        for v in range(Nv):
            for t in range(Nt):
                best = 1e10
                for sy in range(Ry):
                    for sz in range(Rz):
                        m = np.zeros((Ny, Nz))
                        m[sy::Ry, sz::Rz] = 1
                        d = np.abs(m - km[:, :, v, t]).sum()
                        if best > d:
                            best = d
                            shift[v, t] = (sy, sz)
        return shift

    @classmethod
    def acs_philips(cls, kspace, Ry, Rz):
        """Port of ``ACS_Philips``. Returns (acs_size, shift [0-based], kspace_ACS)."""
        k6 = _first_page(kspace, 6)
        Nx = k6.shape[0]
        sampled_all = (sos(k6[matlab_mid(Nx)], axis=2) != 0).all(axis=(2, 3))  # [Ny,Nz]
        iy, iz = np.nonzero(sampled_all)
        acs_size = [round(iy.max() - iy.min()) / 2, round(iz.max() - iz.min()) / 2]
        shift = cls.find_shift(k6, Ry, Rz)
        acs_mask = (sos(k6, axis=3) != 0).all(axis=(3, 4))                       # [Nx,Ny,Nz]
        kspace_acs = k6 * acs_mask[:, :, :, None, None, None]
        return acs_size, shift, kspace_acs

    @staticmethod
    def add_acs_siemens(kspace, k_acs, Ry, Rz):
        """Port of ``Add_ACS_Siemens``: paste separately acquired ACS lines into
        the centre of k-space. Returns (kspace copy, acs_size)."""
        a1 = (k_acs.shape[1] - Ry + 1) / 2
        a2 = (k_acs.shape[2] - Rz + 1) / 2
        Ny, Nz = kspace.shape[1:3]
        out = np.array(kspace, copy=True)
        out[:, mslice(Ny / 2 - a1 + 1, Ny / 2 + a1), mslice(Nz / 2 - a2 + 1, Nz / 2 + a2)] = \
            k_acs[:, :int(2 * a1), :int(2 * a2)]
        return out, [a1, a2]

    # ------------------------------------------------------------ undersampling
    @staticmethod
    def undersample_kspace(kspace, Ry, Rz, acs_size, interleaving):
        """Port of ``Undersample_kspace``.

        interleaving: 0 none, 1 shift ky with venc and kz with time,
        2 same lattice positions but data taken from the unshifted lines.
        Returns (k_R, ACS).
        """
        Nx, Ny, Nz, Nc, Nv, Nt = kspace.shape
        k_R = np.zeros_like(kspace)
        base_y, base_z = np.arange(0, Ny, Ry), np.arange(0, Nz, Rz)
        for v in range(Nv):
            for t in range(Nt):
                if interleaving == 1:
                    sy, sy2, sz, sz2 = v % Ry, v % Ry, t % Rz, t % Rz
                elif interleaving == 2:
                    sy, sy2, sz, sz2 = v % Ry, 0, t % Rz, 0
                else:
                    sy = sy2 = sz = sz2 = 0
                ky = base_y[base_y + sy < Ny]
                kz = base_z[base_z + sz < Nz]
                dst = k_R[:, :, :, :, v, t]
                src = kspace[:, :, :, :, v, t]
                dst[:, (ky + sy)[:, None], (kz + sz)[None, :], :] = \
                    src[:, (ky + sy2)[:, None], (kz + sz2)[None, :], :]
        ys = mslice(Ny / 2 - acs_size[0] + 1, Ny / 2 + acs_size[0] + 1)
        zs = mslice(Nz / 2 - acs_size[1] + 1, Nz / 2 + acs_size[1] + 1)
        k_R[:, ys, zs] = kspace[:, ys, zs]
        return k_R, np.array(kspace[:, ys, zs], copy=True)

    @classmethod
    def undersample_comp_siemens_mvenc(cls, k_compose, k_acs, Ry, Rz, caipi):
        """Port of ``Undersample_comp_Siemens_mVENC``."""
        k_compose, acs_size = cls.add_acs_siemens(k_compose, k_acs, Ry, Rz)
        return cls.undersample_comp_mvenc(k_compose, Ry, Rz, acs_size, caipi)

    @classmethod
    def undersample_comp_mvenc(cls, k_compose, Ry, Rz, acs_size, caipi, shift_calc=None):
        """Port of ``Undersample_comp_mVENC``. Returns (k_R, ACS, mask_YZ, NetKTV2).

        ``caipi`` is accepted for compatibility; as in the MATLAB code the CAIPI
        shift is computed but not used (the shifted lines are commented out).
        shift_calc: optional [Nv,Nt,2] 0-based shifts (e.g. from find_shift).
        """
        Nx, Ny, Nz, Nc, Nv, Nt = k_compose.shape
        k_R = np.zeros_like(k_compose)
        mask_YZ = np.zeros((Nx, Ny, Nz, Nv, Nt))
        xm = matlab_mid(Nx)

        def initial_shift(v, t):
            if shift_calc is not None:
                return int(shift_calc[v, t, 0]), int(shift_calc[v, t, 1])
            # first non-zero in MATLAB (column-major) order
            nz = np.argwhere(k_compose[xm, :, :, 0, v, t].T != 0)
            if nz.size == 0:
                raise ValueError(f"frame (v={v}, t={t}) is empty at x = end/2")
            return int(nz[0, 1]), int(nz[0, 0])

        shifts = {(v, t): initial_shift(v, t) for v in range(Nv) for t in range(Nt)}
        for (v, t), (iy, iz) in shifts.items():
            k_R[:, iy::Ry, iz::Rz, :, v, t] = k_compose[:, iy::Ry, iz::Rz, :, v, t]

        ys = mslice(Ny / 2 - acs_size[0] + 1, Ny / 2 + acs_size[0] + 1)
        zs = mslice(Nz / 2 - acs_size[1] + 1, Nz / 2 + acs_size[1] + 1)
        k_R[:, ys, zs] = k_compose[:, ys, zs]
        ACS = np.array(k_compose[:, ys, zs], copy=True)

        cK = 0
        for cZ in range(Rz):
            for cY in range(Ry):
                for (v, t), (iy, iz) in shifts.items():
                    mask_YZ[:, iy + cY::Ry, iz + cZ::Rz, v, t] = cK
                cK += 1
        net_ktv2 = cls._assemble_net(Ry, Rz, Nc, cls.NetT, cls.NetVenc)

        mask_acs = np.all(k_R != 0, axis=(3, 4, 5))            # [Nx,Ny,Nz]
        mask_YZ = mask_YZ * (1 - mask_acs[..., None, None])
        return k_R, ACS, mask_YZ, net_ktv2

    # ------------------------------------------------------------ kernel networks
    @staticmethod
    def _net_yz(cY, cZ, Ry, Rz):
        return [(-cY, -cZ), (-cY, Rz - cZ), (Ry - cY, -cZ), (Ry - cY, Rz - cZ)]

    @classmethod
    def _assemble_net(cls, Ry, Rz, Nc, net_t, net_v, with_vt_index=False):
        """Build NetKTV [K, 4 (or 6), Ry*Rz]: columns dx, dy, dz, coil (+ iv, it)."""
        rows_per_k = len(cls.NetX) * 4 * len(net_v) * len(net_t) * Nc
        net = np.zeros((rows_per_k, 6 if with_vt_index else 4, Ry * Rz), dtype=np.int64)
        cK = 0
        for cZ in range(Rz):
            for cY in range(Ry):
                if cK > 0:
                    rows = []
                    for dx in cls.NetX:
                        for dy, dz in cls._net_yz(cY, cZ, Ry, Rz):
                            for iv, dv in enumerate(net_v):
                                for it, dt in enumerate(net_t):
                                    for c in range(Nc):
                                        r = [dx, dy + dv, dz + dt, c]
                                        if with_vt_index:
                                            r += [iv, it]
                                        rows.append(r)
                    net[:, :, cK] = rows
                cK += 1
        return net

    @classmethod
    def _mode_nets(cls, Ry, Rz, mode, truncate=True):
        if mode == 0:
            return cls.NetT0, cls.NetVenc0
        nt = cls.NetT[:min(len(cls.NetT), Rz)] if truncate else cls.NetT[:Rz]
        if mode == 1:
            return nt, cls.NetVenc0
        nv = cls.NetVenc[:min(len(cls.NetVenc), Ry)] if truncate else cls.NetVenc[:Ry]
        return nt, nv

    @classmethod
    def get_net_ktv(cls, Ry, Rz, Nc, mode):
        """Port of ``getNetKTV`` (mode 0 GRAPPA, 1 k-t GRAPPA, 2 k-t-v GRAPPA)."""
        net_t, net_v = cls._mode_nets(Ry, Rz, mode)
        return cls._assemble_net(Ry, Rz, Nc, net_t, net_v)

    @classmethod
    def get_net_ktv2(cls, Ry, Rz, Nc, mode):
        """Port of ``getNetKTV2``: adds columns 4/5 = 0-based venc / time offset index."""
        net_t, net_v = cls._mode_nets(Ry, Rz, mode, truncate=False)
        return cls._assemble_net(Ry, Rz, Nc, net_t, net_v, with_vt_index=True)

    @classmethod
    def get_net_ktv3(cls, Ry, Rz, Nc, mode=None):
        """Port of ``getNetKTV3``: dense ky/kz block, ``mode`` unused (as in MATLAB)."""
        rows = [[dx, dy, dz, c]
                for dx in cls.NetX
                for dy in range(-(Ry - 1), Ry + 1)
                for dz in range(-(Rz - 1), Rz + 1)
                for c in range(Nc)]
        net = np.zeros((len(rows), 4, Ry * Rz), dtype=np.int64)
        for cK in range(1, Ry * Rz):
            net[:, :, cK] = rows
        return net

    # ------------------------------------------------------------ coil maps
    @staticmethod
    def espirit_km(raw, w=5, thresh=0.01):
        """Port of ``ESPIRIT_KM`` (local-eigenvector coil maps).

        raw [nx, ny, nc, nz] coil images -> maps [nx, ny, nc, nz] (same layout
        as the MATLAB function).
        """
        data = np.transpose(raw, (0, 1, 3, 2))               # [Nx,Ny,Nz,Nc]
        S, M = _local_eig_maps(data, w)
        coil = S * (M > thresh * np.abs(M).max())[..., None]
        return np.transpose(coil, (0, 1, 3, 2))

    @staticmethod
    def espirit_km_2d(raw_nz, w=5, thresh=0.01):
        """Port of ``ESPIRIT_KM_2D``: slice-by-slice maps. raw_nz [Nx,Ny,Nz,Nc]."""
        coil = np.zeros(raw_nz.shape, dtype=np.complex128)
        for z in range(raw_nz.shape[2]):
            S, M = _local_eig_maps(raw_nz[:, :, z, :], w)
            coil[:, :, z, :] = S * (M > thresh * np.abs(M).max())[..., None]
        return coil