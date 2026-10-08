
load("C:/Users/kevin/Downloads/test_kspace.mat")

t0=tic;
% Example on a fully sampled k-space undersampled with Ry=3 and Rz=3, for a prospective undersample case use undersample=0
% By default the CUDA mode is 1 (faster recon), make sure to have installed and compiled the CUDA code, otherwise use the slow CPU mode
[k_recon]=Undersample_n_recon_KTV_GRAPPA(kspace,[3 3],1,2,[],[],1,8,1e-3,1,1);

disp(['R=(3,3) mode=2 (' num2str(toc(t0)) 's)'])

save("C:/Users/kevin/Downloads/kspace_matlab.mat",'k_recon')