load('test_data_CPP.mat')


%%% Recon Matlab only (VERY SLOW)
tic; 
k_recon_matlab=GRAPPA_5D_ktv.Recon_n_Train_5D (k_R_composite_gpu,k_input_gpu(:,:,:,:,2,2),ACS_composite_gpu,ACS_gpu(:,:,:,:,2,2),mask_YZ(:,:,:,2,2),NetKTV,cptRy,cptRz);
disp(['time to recon one 3D volume on matlab ' num2str(toc)]);

%%% Recon CUDA only (FAST)
tic; 

k_recon_GPU = Recon_n_Train_5D_2026_2(k_R_composite_gpu,k_input_gpu(:,:,:,:,2,2),ACS_composite_gpu,ACS_gpu(:,:,:,:,2,2),mask_YZ(:,:,:,2,2),int32(NetKTV),cptRy,cptRz); % convert NetKTV to Int, very important
k_recon_CUDA=gather(k_recon_GPU);
 
disp(['time to recon one 3D volume on CUDA ' num2str(toc)]);