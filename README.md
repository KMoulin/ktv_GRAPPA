# Implementation of the 5D ktv GRAPPA reconstruction

- script_test_ktv_GRAPPA.m -> Script example of ktv GRAPPA reconstruction on an undersampled dataset (dataset available here https://github.com/KMoulin/ktv_GRAPPA/releases/tag/v0.0)

- script_undersample_ktv_GRAPPA.m -> Script to undersample a k-space and run a GRAPPA reconstruction on a full 4D flow dataset

- GRAPPA_5D_ktv.m -> static library containing the main function of the ktv GRAPPA reconstruction implemented in Matlab, needs to be added to the path

- CUDA/Recon_n_Train_5D_2026_2_compile_mex.m -> Script to compile the MEXCUDA source code

- CUDA/Recon_n_Train_5D_2026_2.cu -> CUDA implementation of the ktv GRAPPA reconstruction, needs to be added to the path

# Dependencies

## Matlab
- SPIRIT v0.3 for coil compression -> https://people.eecs.berkeley.edu/~mlustig/Software.html

## CUDA
- CUDA Toolkit version V11+ (tested here with 11.8 and 13.0)
- A mex compiller compatible with CUDA (here Microsoft Visual C++ 2022 (C))
