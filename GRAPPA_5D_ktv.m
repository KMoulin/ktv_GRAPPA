classdef GRAPPA_5D_ktv

    properties (Constant)
        NetX=[2 1 0 -1 -2];
        NetT=[-1 0 1];
        NetVenc=[-1 0 1];
        NetT0=[0];
        NetVenc0=[0];
    end
    methods(Static)
        
         function k_recon=Recon_n_Train_5D (k_composite_gpu,k_R_gpu,ACS_composite_gpu,ACS_gpu,mask_R,NetKTV,Ry,Rz)
    
            k_recon_gpu=gpuArray(single(k_R_gpu));

            [Nx Ny Nz Nc Nvenc Ncardiac]=size(k_R_gpu);

            h=waitbar(0,'nkernel GRAPPA 5D');
            for nt = 1:1:1 % Time dimension is managed outside of the function 
                for nv = 1:1:1 % Velocity dimension is managed outside of the function
                    cK=0;
                    for cZ=0:1:Rz-1
                        for cY=0:1:Ry-1
                            if cK>0

                                %%% Train from ACS

                                % We gather all the example possible and put the example in list_train which is a double matrix of size [Example, 3D [x y z]]
                                list_train=GRAPPA_5D_ktv.getListofExample(ACS_gpu,NetKTV(:,:,cK));
                                
                                
                                
                                % Sn_GPU is a matrix of points surrouding the acquired points, here we gather them from the composite space. GPU single complex matrix of size [(NCoil x Patch size) x Example]
                                [Sn_GPU]=GRAPPA_5D_ktv.Patch_GPU(ACS_composite_gpu,list_train,NetKTV(:,:,cK));  

                                % Xn_GPU is a matrix of the acquired points. GPU single complex matrix of size [Ncoil x Example]
                                [Xn_GPU]=GRAPPA_5D_ktv.Patch_GPU_ACS(ACS_gpu,list_train);
                               
                                % Wn_GPU is the GRAPPA weight matrix. GPU single complex matrix of size[(NCoil x Patch size) x NKernel]
                                Wn_GPU=((Xn_GPU * Sn_GPU') / (Sn_GPU * Sn_GPU'));

                                %%% clear GPU Memory
                                clear Sn_GPU Xn_GPU list_train;

                                %%% Recon from data

                                % We create a list of point index based on
                                % the current kernel number cK.
                                [idx]=find(mask_R==cK);
                                [xidx,yidx,zidx] = ind2sub([Nx Ny Nz],idx);
                                list_recon=[xidx yidx zidx]; % 40K for R12, 1.8M for R3

                                lx=length(NetKTV)* Nc;
                                ly=size(list_recon,1);
                               
                                 %%% Check the memory and recon
                                 % We don't want to load more of 5Gb into the GPU 
                                 % if we use more, we recontruct only a subset of points at the time from list_recon which is a matrix of size [Points, 3D [x y z]]                 
                                if (lx*ly*16/(1024^3)>4) % GBytes of S
                                    segment=ceil(lx*ly*16/(1024^3)/4);
                                    block_size=round(ly/segment);
                                    for cpt_seg=1:1:segment
                                         st=(block_size*(cpt_seg-1)+1);
                                         ed=min(block_size*(cpt_seg),ly);
                                         VectorIdx=[st:1:ed];
                                         % we have a sub list of points to recon
                                         list_recon_sub=list_recon(VectorIdx,:);

                                        % Sr_GPU is a matrix of points surrouding the missing points, here we gather them from the composite space. GPU single complex matrix of size [(NCoil x Patch size) x Recon points sub]
                                        [Sr_GPU]=GRAPPA_5D_ktv.Patch_GPU(k_composite_gpu,list_recon_sub,NetKTV(:,:,cK));
                                        
                                        % Xr_GPU is a matrix of the GRAPPA reconstructed points. GPU single complex matrix of size [Ncoil x Recon points sub] 
                                        Xr_GPU=Wn_GPU*Sr_GPU;

                                        % Convert 3D subscripts to linear indices once
                                        linear_idx = sub2ind(size(k_recon_gpu),list_recon_sub(:,1),list_recon_sub(:,2), list_recon_sub(:,3));

                                        % Expand indices for the 4th dimension (coils)
                                        Nc = size(k_recon_gpu, 4);  % number of coils
                                        linear_idx_expanded = repmat(linear_idx, 1, Nc) + ...
                                                              (0:Nc-1) * numel(k_recon_gpu(:,:,:,1));

                                        % Assign all at once
                                        k_recon_gpu(linear_idx_expanded) = Xr_GPU.';  % Note transpose
                                    end
                                else

                                    % Sr_GPU is a matrix of points surrouding the missing points, here we gather them from the composite space. GPU single complex matrix of size [(NCoil x Patch size) x Recon points]
                                    [Sr_GPU]=GRAPPA_5D_ktv.Patch_GPU(k_composite_gpu,list_recon,NetKTV(:,:,cK));

                                    % Xr_GPU is a matrix of the GRAPPA reconstructed points. GPU single complex matrix of size [Ncoil x Recon points] 
                                    Xr_GPU=Wn_GPU*Sr_GPU;

                                    % Convert 3D subscripts to linear indices once
                                    linear_idx = sub2ind(size(k_recon_gpu),list_recon(:,1),list_recon(:,2),list_recon(:,3));
                                    
                                    % Expand indices for the 4th dimension (coils)
                                    Nc = size(k_recon_gpu, 4);  % number of coils
                                    linear_idx_expanded = repmat(linear_idx, 1, Nc) + ...
                                                          (0:Nc-1) * numel(k_recon_gpu(:,:,:,1));
                                    
                                    % Assign all at once
                                    k_recon_gpu(linear_idx_expanded) = Xr_GPU.';  % Note transpose
                                end
                            end
                            cK=cK+1;
                            waitbar(cK/(Rz*Ry),h)
                        end

                    end
                end
            end
           
            k_recon=gather(k_recon_gpu(:,:,:,:,1,1)); % From GPU to RAM
            close(h)
         end
 function data_GPU = Patch_GPU(vect_mat,list_points,NetKTV)
            % Convert 3D subscripts to linear index
            dims = size(vect_mat);
           
            % Initialize output with zero
            data_GPU = gpuArray(single(zeros(size(NetKTV, 1),size(list_points, 1))));

            for cpt=1:1:size(list_points,1)
                 list=list_points(cpt,:)+NetKTV(:,1:3);
                 % Check which points are valid
                 isValid = all(list >= 1 & list <= dims(1:3), 2);  
                  
                 linIdx = sub2ind(size(vect_mat), list_points(cpt,1)+NetKTV(isValid,1), list_points(cpt,2)+NetKTV(isValid,2),list_points(cpt,3)+NetKTV(isValid,3), NetKTV(isValid,4));
                 % Extract data
                 data_GPU(isValid,cpt) = vect_mat(linIdx);
            end
           
 end   
 
  function data_GPU = Patch_GPU_ACS(vect_mat,list_points)
            % Convert 3D subscripts to linear index                
            dims = size(vect_mat);
           
            % Initialize output with zeros
            data_GPU = gpuArray(single(zeros(dims(4),size(list_points,1))));

           for cC=1:1:dims(4)
                linIdx = sub2ind(size(vect_mat), list_points(:,1), list_points(:,2),list_points(:,3),ones(size(list_points,1),1)*cC);
                % Extract data
                data_GPU(cC,:) = vect_mat(linIdx);
           end
           
      end     
 
 
function list_pts=getListofExample(ACS,NetKTV)
            % ACS is 6D [Nx, Ny, Nz, Nc, Nv, Nt];
            [Nx, Ny, Nz]=size(ACS);

            % Right before calling getListofExample2026:
            % fprintf('=== MATLAB DEBUG ===\n');
            % fprintf('ACS size: [%d %d %d]\n', size(ACS, 1), size(ACS, 2), size(ACS, 3));
            % fprintf('NetKTV(:,:,cK=%d) size: [%d %d]\n', 1, size(NetKTV, 1), size(NetKTV, 2));
            % fprintf('NetKTV bounds:\n');
            % fprintf('  X: [%d, %d]\n', min(NetKTV(:,1,1)), max(NetKTV(:,1,1)));
            % fprintf('  Y: [%d, %d]\n', min(NetKTV(:,2,1)), max(NetKTV(:,2,1)));
            % fprintf('  Z: [%d, %d]\n', min(NetKTV(:,3,1)), max(NetKTV(:,3,1)));

            list_pts=[];
            cpt=1;
            for cpt_x=1:1:size(ACS,1)
                for cpt_y=1:1:size(ACS,2)
                    for cpt_z=1:1:size(ACS,3)
                        if (cpt_x+min(NetKTV(:,1)))>0 & (cpt_x+max(NetKTV(:,1)))<=Nx
                            if (cpt_y+min(NetKTV(:,2)))>0 & (cpt_y+max(NetKTV(:,2)))<=Ny
                                if (cpt_z+min(NetKTV(:,3)))>0 & (cpt_z+max(NetKTV(:,3)))<=Nz
                                    list_pts(cpt,:)=[cpt_x,cpt_y,cpt_z];
                                    cpt=cpt+1;
                                end
                            end
                        end
                    end
                end
            end
end

      
        function [k_R,ACS,mask_YZ,NetKTV2]=Undersample_comp_mVENC(k_compose,Ry,Rz,acs_size,caipi)

            [Nx Ny Nz Nc Nvenc Ncardiac]=size(k_compose);
            k_R=zeros(size(k_compose));
            mask_YZ=zeros([Nx Ny Nz]);
            ACS=zeros([Nx acs_size(1)*2 acs_size(2)*2 Nc Nvenc Ncardiac]);
            mask_R2=mask_YZ;
            mask_R2(1:Ry:end,1:Rz:end)=1;

            Sv=Ry;
            Sc=Rz;
            shiftY=0;
            shiftZ=0;
            
            % Undersample the data with or without shift
            for cptY=1:Ry:(Ny)
                for cptZ=1:Rz:(Nz)
                    for cptV=1:1:Nvenc
                        if caipi
                            shiftY=mod(cptV-1,Sv);
                        end
                        for cptCar=1:1:Ncardiac
                            if caipi
                                shiftZ=mod(cptCar-1,Sc);
                            end

                            if cptY+shiftY<=Ny && cptZ+shiftZ<=Nz
                                k_R(:,cptY+shiftY,cptZ+shiftZ,:,cptV,cptCar)=k_compose(:,cptY+shiftY,cptZ+shiftZ,:,cptV,cptCar);
                            end

                        end
                    end
                end
            end

            % Add the center lines as ACS lines
            k_R(:,end/2-acs_size(1):end/2+acs_size(1),end/2-acs_size(2):end/2+acs_size(2),:,:,:)=k_compose(:,end/2-acs_size(1):end/2+acs_size(1),end/2-acs_size(2):end/2+acs_size(2),:,:,:);
            ACS=k_compose(:,end/2-acs_size(1):end/2+acs_size(1),end/2-acs_size(2):end/2+acs_size(2),:,:,:);

            % create a mask that map each kernel and the network of points
            % the points already acquired correspond to the kernel 0
            cK=0;
            for cZ=0:1:Rz-1
                for cY=0:1:Ry-1
                    for cptV=1:1:Nvenc
                        if caipi
                            shiftY=mod(cptV-1,Sv);
                        end
                        for cptCar=1:1:Ncardiac
                            if caipi
                                shiftZ=mod(cptCar-1,Sc);
                            end
                           
                            mask_YZ(:,cY+1+shiftY:Ry:end,cZ+1+shiftZ:Rz:end,cptV,cptCar)=cK;
                            
                        end

                    end

                   
                    % create the network of point corresponding to ktv-GRAPPA
                    NetX=GRAPPA_5D_ktv.NetX;
                    NetT=GRAPPA_5D_ktv.NetT;
                    NetVenc=GRAPPA_5D_ktv.NetVenc;
                    if cK>0
                        NetYZ(:,:,cK)=[0-cY 0-cZ;0-cY Rz-cZ;Ry-cY 0-cZ; Ry-cY Rz-cZ];
                        cpt=1;
                        for cpt_x=1:1:length(NetX)
                            for cpt_yz=1:1:size(NetYZ,1)
                                for cpt_v=1:1:length(NetVenc)
                                    for cpt_t=1:1:length(NetT)
                                        for cC=1:Nc 
                                              NetKTV2(cpt,1,cK)= NetX(cpt_x);
                                              NetKTV2(cpt,2,cK)=NetYZ(cpt_yz,1,cK)+NetVenc(cpt_v);
                                              NetKTV2(cpt,3,cK)=NetYZ(cpt_yz,2,cK)+NetT(cpt_t);
                                              NetKTV2(cpt,4,cK)=cC;
                                              cpt=cpt+1;
                                              
                                        end
                                    end
                                end
                            end
                        end           
                    end
                    cK=cK+1;
                end
            end
            % all the ACS lines have a kernel of 0
            mask_YZ(:,end/2-acs_size(1):end/2+acs_size(1),end/2-acs_size(2):end/2+acs_size(2),:,:)=0;
        end

        function coil=ESPIRIT_KM(Raw)
        % 
        % [nx ny nc] = size(Raw);
        % m2 = reshape(Raw, nx * ny, nc);
        % [ev ed] = eig(m2' * m2);
        % ecmap = reshape(m2 * ev, nx, ny, nc);
        % energy = (ed / max(ed(:))) * 100;
        % coil = flip(ecmap,3);
        
        % Raw [nx ny nc nz]
            data_2d(:,:,:,:)=permute(Raw,[1 2 4 3]);
            [Nx,Ny,Nz,Nc] = size(data_2d);
            S = zeros(Nx,Ny,Nz,Nc);
            M = zeros(Nx,Ny,Nz);
            w = 5;
            for i = 1:Nx
                ii = max(i-w,1):min(i+w,Nx);
                for j = 1:Ny
                    jj = max(j-w,1):min(j+w,Ny);
                    for k = 1:Nz
                        kk = max(k-w,1):min(k+w,Nz);
                        kernel = reshape(data_2d(ii,jj,kk,:),[],Nc);
                        [V,D] = eigs(conj(kernel'*kernel),1);
                        S(i,j,k,:) = V*exp(-1j*angle(V(1)));
                        M(i,j,k) = sqrt(D);
                    end
                end
            end
            coil = permute(squeeze(S.*(M>0.01*max(abs(M(:))))),[1 2 4 3]);
        end
    end
end

