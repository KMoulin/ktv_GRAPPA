# Implementation of ktv GRAPPA reconstruction

Main repository for the ktv GRAPPA implementation for 4D flow. Implementation as a universal GRAPPA fashion working with any Patch shape (retro compatible with GRAPPA and k-t GRAPPA). Reconstruct any kspace under the format of a 6D complex matrix [X Y Z Coils Velocities/Venc Time] with a linear interleaving pattern. ktv GRAPPA was mainly developed for 4D flow reconstruction but can theoretically work with any 6D kspace data (multi-contrast/multi-time points). 

ktv GRAPPA was developed on Matlab and converted to Python using AI. For reference implementation please refer to the Matlab version. 

A fully sampled kspace dataset is available here (TODO)

For performance please use the GPU/CUDA options (~1 minute) against CPU (~1 hour) when possible.

Tested on Matlab 2021B, 2023B, Python 3.13.12 with CUDA 13.4 on a NVIDIA GeForce RTX 4060 with NVDIA-SMI Driver 617.42

## Matlab (from authors)
### Source code
- Matlab/test_ktv_GRAPPA.m -> Script example of ktv GRAPPA reconstruction
- Matlab/Undersample_n_recon_KTV_GRAPPA.m ->  Main function to manage reconstructions of the ktv GRAPPA, needs to be added to the path
- Matlab/GRAPPA_5D_ktv.m -> static library containing the code of ktv GRAPPA reconstruction implemented in Matlab, needs to be added to the path
### Dependencies
- SPIRIT v0.3 for coil compression and fft3c -> https://people.eecs.berkeley.edu/~mlustig/Software.html

## CUDA/Matlab (from authors, optimized using AI)
### Source code
- CUDA/Recon_n_Train_5D_2026_3_compile_mex.m -> Script to compile the MEXCUDA source code
- CUDA/Recon_n_Train_5D_2026_3.cu -> CUDA implementation of the ktv GRAPPA reconstruction, needs to be added to the path
### Dependencies
- CUDA Toolkit version V11+ (tested here with 11.8 and 13.0)
- A mex compiller compatible with CUDA (here Microsoft Visual C++ 2022 (C))

## Python (converted from Matlab using AI)
### Source code
- Python/test_ktv_GRAPPA.py -> Script example of ktv GRAPPA reconstruction
- Python/undersample_n_recon_ktv_grappa.py ->  Main function to manage reconstructions of the ktv GRAPPA, needs to be added to the path
- Python/grappa_5D_ktv.py -> static library containing the code of ktv GRAPPA reconstruction implemented in Python, needs to be added to the path
- coil_utils.py -> Dependencies functions to match Matlab implementation
### Dependencies
cupy-cuda13x==14.2.0
h5py==3.16.0
matplotlib==3.11.1
numpy==2.4.4
scipy==1.17.1

