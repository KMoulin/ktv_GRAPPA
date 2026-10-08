import time

import numpy as np

from coil_utils import fft3c, sos
from grappa_5d_ktv import GRAPPA5DKtv
from undersample_n_recon_ktv_grappa import undersample_n_recon_ktv_grappa

from scipy.io import loadmat

import h5py



def load_mat73_complex(path, varname):
    with h5py.File(path, "r") as f:
        raw = f[varname][()]          # reads the whole dataset into memory
    if raw.dtype.names and {"real", "imag"} <= set(raw.dtype.names):
        data = raw["real"] + 1j * raw["imag"]
    else:
        data = raw                    # variable was actually real
    # Reverse axis order to match MATLAB indexing; make it C-contiguous
    return np.ascontiguousarray(data.transpose())

def save_mat73_complex(path, varname, A):
    A = np.asarray(A, dtype=np.complex128).transpose()   # back to MATLAB axis order
    out = np.empty(A.shape, dtype=[("real", "<f8"), ("imag", "<f8")])
    out["real"], out["imag"] = A.real, A.imag

    with h5py.File(path, "w", userblock_size=512) as f:
        d = f.create_dataset(varname, data=out, compression="gzip")
        d.attrs["MATLAB_class"] = np.bytes_("double")

    # MAT-file header in the 512-byte userblock:
    # 116 bytes text, 8 bytes subsys offset, version 0x0200, endian "IM"
    text = b"MATLAB 7.3 MAT-file, Platform: GLNXA64, Created by: h5py HDF5 schema 1.00 ."
    hdr = text.ljust(116, b" ") + b"\x00" * 8 + (0x0200).to_bytes(2, "little") + b"IM"
    with open(path, "r+b") as fh:
        fh.write(hdr.ljust(512, b"\x00"))


complex_matrix = load_mat73_complex("C:/Users/kevin/Downloads/Patient6/kspace.mat", "kspace")   # "A" = your MATLAB variable name
print(complex_matrix.shape, complex_matrix.dtype)                       # e.g. (a, b, c, d, e, f) complex128

t0 = time.perf_counter()
k_rec = undersample_n_recon_ktv_grappa(complex_matrix, (3,3), mode=2, undersample=0, nc_cc=8, verbose=True,use_gpu=True,coil_sens=True,lam=1e-3)
print(f"R={(3,3)} mode={2} ({time.perf_counter() - t0:.1f} s)")


save_mat73_complex("C:/Users/kevin/Downloads/Patient6/kspace_python.mat", "k_rec", k_rec)
