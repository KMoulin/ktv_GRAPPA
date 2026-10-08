
function [img_recon_all2]=Undersample_n_recon_KTV_GRAPPA(kspace,listR,Channel_Sens_full,mode,listV,listT,undersample,nc_CC,lambda,force_sampling_mask,ridge_Tikhonov)
%% Recon parameters INPUT
% 'kspace' (Single or Double Complex Matrix) is a 6D matrix complex of dim [x y z coil Velocity Time] 
% 'listR' (Double Vector) contains the acceleration factors [Ry, Rz]
% 'Channel_Sens_full' (Single or Double Complex Matrix or Boolean) is a 4D matrix complex of dim [x y z coil]. If empty, reconstructed kspace is return instead of the image. If yes, use ESPRIT to do coil sensitivity maps from ACS lines.
% 'mode', (double) 0 for GRAPPA, 1 for k-t GRAPPA and 2 for k-t-v GRAPPA
% 'listV' (double vector) contains the indices for the Velocity dimension, i.e.[1, 2, 4, 12, 13, 14], is left empty [] all the Velocity indices of kspace are reconstructed
% 'listV' (double vector) contains the indices for the Time/Cardiac dimension, i.e.[1, 2, 4, 12, 13, 14], is left empty [] all the Cardiac indices of kspace are reconstructed
% 'undersample' (boolean) if yes, undersample 'kspace' at the rate of 'listR' (for instance if 'kspace' is fully sampled)
% 'nc_CC' (double) number of coil/channel we want in the coil compression, if left empty [] no coil compression is used. Usually 8
% 'lambda' (double, %) for Tikhonov regularization (usually 1e-3). If left empty [] no regularization is used
% 'force_sampling_mask' (bool) impose the linear sampling mask on top of the automatically detected mask. 0 for most cases
% 'ridge_Tiknonov' (not used)
%% Recon parammeter OUTPUT
% 'img_recon_all2' (Single or Double Complex Matrix) is a 5D matrix complex of dim [x y z Velocity Time] if Channel_Sens_full is given or equal to 1
% 'img_recon_all2' (Single or Double Complex Matrix) is a 6D matrix complex of dim [x y z coil Velocity Time] if Channel_Sens_full is empty
%
%% Example
% Example of usage for ktv-GRAPPA:
% [img_recon]=Undersample_n_recon_KTV_GRAPPA(kspace,[3 3],1,3); 
%
% Example of usage for ktv-GRAPPA (kspace only);
% [kspace_recon]=Undersample_n_recon_KTV_GRAPPA(kspace,[3 3],[],3);
%
kspace(isinf(kspace))=0;
Tmax=size(kspace,6);
Vmax=size(kspace,5);
CCmax=size(kspace,4);
Coil_combine=1;

if nargin<3 || isempty(Channel_Sens_full)
   Coil_combine=0; 
end
if nargin<4 || isempty(mode)
   mode=2; % Ktv-GRAPPA recon 
end
if nargin<5 || isempty(listT)
    listT=1:1:Tmax;
end
if nargin <6 || isempty(listV)
    listV=1:1:Vmax;
end
if nargin<7 || isempty(undersample)
   undersample=0; 
end
if nargin<8 || isempty(nc_CC)
   nc_CC=CCmax;
end
if nargin<9 || isempty(lambda)
   lambda=0;
end
if nargin<10 || isempty (force_sampling_mask)
    force_sampling_mask=0;
end
if nargin<11 || isempty (ridge_Tikhonov)
    ridge_Tikhonov=0;
end
cptRy=listR(1); % Acceleration factor in Y
cptRz=listR(2); % Acceleration factor in Z


%% Undersample kspace
if undersample~=0
    [kspace,~]=GRAPPA_5D_ktv.Undersample_kspace(kspace,cptRy,cptRz,[10 8],undersample);
end

mask_0=(abs(mean(kspace,4)));
mask_0(mask_0~=0)=1;
mask=squeeze(min(mask_0(round(end/2),:,:,:,:,:),[],[4 5 6]));
[idy, idz]=find(mask);
ACS_size=[(max(idy)-min(idy))/2,(max(idz)-min(idz))/2];

%% Coil compression and corresponding coil sensitivity
% Requiere the Espirit package 0.3

if nc_CC~=CCmax
    tic
    dim_CC=1;
    kspace_CC=zeros(size(kspace(:,:,:,1:nc_CC,:,:)));
    calib=permute(kspace(:,end/2-ACS_size(1):end/2+ACS_size(1),end/2-ACS_size(2):end/2+ACS_size(2),:,1,1),[1 2 3 4 5 6]);
    gccmtx = calcGCCMtx(calib,dim_CC,5);
    gccmtx_aligned = alignCCMtx(gccmtx(:,1:nc_CC,:));
    for cpt_fd=1:1:size(kspace,5)
        for cpt_cp=1:1:size(kspace,6)
            DATAc=permute(kspace(:,:,:,:,cpt_fd,cpt_cp),[1 2 3 4 5 6]);
            kspace_CC(:,:,:,:,cpt_fd,cpt_cp)= CC(DATAc,gccmtx_aligned,dim_CC);
        end
    end
    if Coil_combine
        img_xby2=fftshift(fftshift(fft3c(squeeze(kspace_CC)),2),3);
        Channel_Sens_full=permute(GRAPPA_5D_ktv.ESPIRIT_KM(squeeze(permute(img_xby2(:,:,:,:,1,1),[ 1 2 4 3]))),[1 2 4 3]);
        clear img_xby2;
    end
    CCmax=nc_CC;
    kspace=kspace_CC;
    clear kspace_CC calib gccmtx gccmtx_aligned DATAc;
    disp(['time to run coil compression / coil sensitivity ' num2str(toc) 's']);
end

%% ACS 
ACS=kspace(:,min(idy):max(idy),min(idz):max(idz),:,:,:);


%% GRAPPA Reconstruction
% For Each Velocities
% We assume 4 encoding points

NetKTV=GRAPPA_5D_ktv.getNetKTV(cptRy,cptRz,CCmax,mode);  % 0 GRAPPA, 1 k-t GRAPPA, 2 k-t-v GRAPPA
[mask_YZ2,list_YZ]=GRAPPA_5D_ktv.Mask_discovery(kspace,cptRy,cptRz);
if force_sampling_mask
    mask_YZ=mask_YZ2;
    mask_YZ(:,min(idy):max(idy),min(idz):max(idz),:,:)=0;
else
    mask_0=1-mask_0;
    mask_YZ=mask_YZ2.*squeeze(mask_0(:,:,:,1,:,:));
end

for cpt_dir=listV

    % For Each Time
    for cpt_cardiac=listT
      
        if cptRy==1&&cptRz==1
            k_recon=squeeze(kspace(:,:,:,:,cpt_dir,cpt_cardiac));
        else

            disp([' Velocity ' num2str(cpt_dir) '/' num2str(Vmax) ' Cardiac phase ' num2str(cpt_cardiac) '/'  num2str(Tmax) '  [Ry=' num2str(cptRy) ';Rz=' num2str(cptRz) ']'])
           
            %%% Create Composite k-space and ACS datasets

            [k_R_composite_gpu]=GRAPPA_5D_ktv.Create_Composite_GPU(kspace,list_YZ,cptRy,cptRz,cpt_dir,cpt_cardiac);
            [ACS_composite_gpu]=GRAPPA_5D_ktv.Create_Composite_GPU(ACS,list_YZ,cptRy,cptRz,cpt_dir,cpt_cardiac);

            k_input_gpu=gpuArray(single(kspace(:,:,:,:,cpt_dir,cpt_cardiac)));
            ACS_input_gpu=gpuArray(single(ACS(:,:,:,:,cpt_dir,cpt_cardiac)));

            %%% Recon Matlab only (VERY SLOW)
            % tic
            % [k_recon]=GRAPPA_5D_ktv.Recon_n_Train_5D (k_R_composite_gpu,k_input_gpu,ACS_composite_gpu,ACS_input_gpu,mask_YZ(:,:,:,cpt_dir,cpt_cardiac),NetKTV,cptRy,cptRz,lambda);
            % disp(['time to recon one 3D volume on matlab ' num2str(toc)]);

            %%% Recon CUDA only (FAST)
            tic;    
            [k_recon_GPU] = Recon_n_Train_5D_2026_3(k_R_composite_gpu,k_input_gpu,ACS_composite_gpu,ACS_input_gpu,double(mask_YZ(:,:,:,cpt_dir,cpt_cardiac)),int32(NetKTV),cptRy,cptRz,double(lambda));
            k_recon=gather(k_recon_GPU);

            disp(['time to recon one 3D volume on CUDA ' num2str(toc)]);
            
        end
         
        %%% 3DFFT + Coil combination
        if Coil_combine
            img_recon=fftshift(fftshift(fft3c(k_recon(:,:,:,:,1,1)),2),3);
            img_recon_coil=sum(squeeze(img_recon).*conj(repmat(Channel_Sens_full,1,1,1,1,size(img_recon,5),size(img_recon,6))),4);
            img_recon_all2(:,:,:,cpt_dir,cpt_cardiac)=img_recon_coil;
        else
            img_recon_all2(:,:,:,:,cpt_dir,cpt_cardiac)=k_recon(:,:,:,:,1,1);
        end
        
    end
end    
end

