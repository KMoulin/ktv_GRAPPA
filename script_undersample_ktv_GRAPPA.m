%% in vivo cine with single kernel training
% Load data
load('Sens_Map.mat')
load('k_cardiac_4D.mat')

%% Recon parameters
% We assume the fully sampled dataset 'kspace' is a 6D matrix complex of dim [x y z coil Velocity Time] 
% 
%
kspace=squeeze(kspace);
kspace(isinf(kspace))=0;
Tmax=size(kspace,6);
Vmax=size(kspace,6);
CCmax=size(kspace,4);

ACS_size=[10 8]; % ky kz ACS
nc_CC=5; % Number of coil compression

cptRy=3; % Acceleration factor in Y
cptRz=3; % Acceleration factor in Z

%% Coil compression and corresponding coil sensitivity
% Requiere the Espirt package 0.3
if nc_CC~=CCmax
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
else
    kspace_CC=kspace;
end
    
img_xby2=fftshift(fftshift(fft3c(squeeze(kspace_CC)),2),3);
Channel_Sens_full=permute(GRAPPA_5D_ktv.ESPIRIT_KM(squeeze(permute(img_xby2(:,:,:,:,1,1),[ 1 2 4 3]))),[1 2 4 3]);

% For Each Velocities
% We assume 4 encoding points
for cpt_dir=1:1:4
    if cpt_dir==1
        vectDir=[2 1 3 4];
    elseif cpt_dir==3
        vectDir=[1 3 2 4];
    elseif cpt_dir==4
        vectDir=[1 4 2 3];
    else 
        vectDir=[1 2 3 4];
    end

    % For Each Time
    for cpt_cardiac=1:1:Tmax
      
        % We take 3 times points
        vectCar=[cpt_cardiac-1 cpt_cardiac  cpt_cardiac+1];
        if cpt_cardiac==1
            vectCar=[Tmax cpt_cardiac  cpt_cardiac+1];
        elseif cpt_cardiac==Tmax
            vectCar=[cpt_cardiac-1 cpt_cardiac  1];
        end
        
        % Only use the 4 Velocities and 3 times points selected
        kspace2=squeeze(kspace_CC(:,:,:,:,vectDir,vectCar));
        if cptRy==1&&cptRz==1
            k_recon=squeeze(kspace_CC(:,:,:,:,vectDir(2),vectCar(2)));
        else

            disp(['sVenc Ry ' num2str(cptRy) ' Rz ' num2str(cptRz) ' venc nb ' num2str(cpt_dir) ' recon type ' num2str(cpt_recon) ' Cine ' num2str(cpt_cardiac) ' CC ' num2str(cpt_cc)])

            %%% Undersample the data, create the mask and the Network of points
            [k_R,mask_R,ACS,mask_YZ,NetG,NetYZ,NetKTV]=GRAPPA_5D_ktv.Undersample_comp_mVENC(squeeze(kspace2),cptRy,cptRz,[10 8],1);
           
            %%% Create Composite k-space and ACS datasets

            % ACS_composite and ACS_gpu are 4D single complex matrix of dim [x ACS_y ACS_z CC]
            ACS_gpu=gpuArray(single(ACS));
            ACS_composite_gpu=gpuArray(single(zeros(size(ACS_gpu(:,:,:,:,1,1)))));
            for cpt_v=1:1:min(size(ACS_gpu,5),cptRy)
                for cpt_t=1:1:min(size(ACS_gpu,6),cptRz)
                        ACS_composite_gpu(:,cpt_v:cptRy:end,cpt_t:cptRz:end,:)=ACS_gpu(:,cpt_v:cptRy:end,cpt_t:cptRz:end,:,cpt_v,cpt_t);
                end
            end
            
            % k_R_composite_gpu and k_input_gpu are 4D single complex matrix of dim [x y z CC]
            k_input_gpu=gpuArray(single(k_R));
            k_R_composite_gpu=gpuArray(single(zeros(size(k_input_gpu(:,:,:,:,1,1)))));
             for cpt_v=1:1:min(size(k_input_gpu,5),cptRy)
                for cpt_t=1:1:min(size(k_input_gpu,6),cptRz)
                        k_R_composite_gpu(:,cpt_v:cptRy:end,cpt_t:cptRz:end,:)=k_input_gpu(:,cpt_v:cptRy:end,cpt_t:cptRz:end,:,cpt_v,cpt_t);
                end
            end

            %%% Recon Matlab only (VERY SLOW)
            %  tic;  
            % k_recon=GRAPPA_5D_ktv.Recon_n_Train_5D_2026_2 (k_R_composite_gpu,k_input_gpu(:,:,:,:,2,2),ACS_composite_gpu,ACS_gpu(:,:,:,:,2,2),mask_YZ(:,:,:,2,2),NetKTV,cptRy,cptRz);
            % disp(['time to recon one 3D volume on matlab ' num2str(toc)]);
            
            %%% Recon CUDA only (FAST)
            tic;    
            k_recon_GPU = Recon_n_Train_5D_2026_2(k_R_composite_gpu,k_input_gpu(:,:,:,:,2,2),ACS_composite_gpu,ACS_gpu(:,:,:,:,2,2),mask_YZ(:,:,:,2,2),int32(NetKTV),cptRy,cptRz);
            k_recon=gather(k_recon_GPU);
            disp(['time to recon one 3D volume on CUDA ' num2str(toc)]);
       
        end
         
        %% 3DFFT + Coil combination
        % k_recon contains the GRAPPA k-space reconstructed kspace 
        img_recon=fftshift(fftshift(fft3c(k_recon(1:size(Channel_Sens_full,1),1:size(Channel_Sens_full,2),1:size(Channel_Sens_full,3),:,1,1)),2),3);
        im_recon_coil_V4=sum(squeeze(img_recon).*conj(repmat(Channel_Sens_full,1,1,1,1,size(img_recon,5),size(img_recon,6))),4);
        img_recon_all2(:,:,:,cpt_dir,cpt_cardiac)=im_recon_coil_V4;
    end
end    
save('img_GRAPPA_Cine_ktv.mat','img_recon_all2');
   

