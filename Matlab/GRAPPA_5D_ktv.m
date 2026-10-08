classdef GRAPPA_5D_ktv

    properties (Constant)
        NetX=[1 0 -1]; % [2 1 0 -1 -2]
        NetT=[0 1 2];    
        NetVenc=[0 1 2 3];
        NetT0=[0];
        NetVenc0=[0];

        MaxVencPts=4;
        MaxTimePts=3;
    end
    methods(Static)
        
         function k_recon=Recon_n_Train_5D (k_composite_gpu,k_R_gpu,ACS_composite_gpu,ACS_gpu,mask_R,NetKTV,Ry,Rz,lambda_scale)
    
            k_recon_gpu=gpuArray(single(k_R_gpu));

            [Nx Ny Nz Nc Nvenc Ncardiac]=size(k_R_gpu);

            h=waitbar(0,'nkernel GRAPPA 5D');

            if isempty(ACS_gpu)
                bExternalW=1;
            else
                bExternalW=0;
            end
            for nt = 1:1:1 % Time dimension is managed outside of the function 
                for nv = 1:1:1 % Velocity dimension is managed outside of the function
                    cK=0;
                    for cZ=0:1:Rz-1
                        for cY=0:1:Ry-1
                            if cK>0

                                %%% Train from ACS
                                if ~bExternalW
                                    % We gather all the example possible and put the example in list_train which is a double matrix of size [Example, 3D [x y z]]
                                    list_train=GRAPPA_5D_ktv.getListofExample(ACS_gpu(:,:,:,1),NetKTV(:,:,cK));
                                    
                                    % Sn_GPU is a matrix of points surrouding the acquired points, here we gather them from the composite space. GPU single complex matrix of size [(NCoil x Patch size) x Example]
                                    [Sn_GPU]=GRAPPA_5D_ktv.Patch_GPU(ACS_composite_gpu,list_train,NetKTV(:,:,cK));  
    
                                    % Xn_GPU is a matrix of the acquired points. GPU single complex matrix of size [Ncoil x Example]
                                    [Xn_GPU]=GRAPPA_5D_ktv.Patch_GPU_ACS(ACS_gpu,list_train);
                                   
                                     % Wn_GPU is the GRAPPA weight matrix. GPU single complex matrix of size[(NCoil x Patch size) x NKernel]
                                     %Wn_GPU=((Xn_GPU * Sn_GPU') / (Sn_GPU * Sn_GPU'));
    
                                    % % With Tikhonov regularization
                                    SSSh = Sn_GPU * Sn_GPU';
                                  
                                    lambda = lambda_scale * trace(SSSh) / size(SSSh, 1);  % 1e-3 ~0.1% of mean diagonal
                                    Wn_GPU = ((Xn_GPU * Sn_GPU') / (SSSh + lambda * eye(size(SSSh))));
                                 
                                    %%% clear GPU Memory
                                    clear Sn_GPU Xn_GPU list_train;
                                else
                                    Wn_GPU=ACS_composite_gpu(:,:,cK);
                                end
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
                                        linear_idx_expanded = repmat(linear_idx, 1, Nc) +  (0:Nc-1) * numel(k_recon_gpu(:,:,:,1));

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
                                    linear_idx_expanded = repmat(linear_idx, 1, Nc) + (0:Nc-1) * numel(k_recon_gpu(:,:,:,1));
                                    
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
        
                    list_pts=[];
                    cpt=1;
                    for cpt_x=1:1:size(ACS,1)
                        for cpt_y=1:1:size(ACS,2)
                            for cpt_z=1:1:size(ACS,3)
                                if (cpt_x+min(NetKTV(:,1)))>0 & (cpt_x+max(NetKTV(:,1)))<=Nx
                                    if (cpt_y+min(NetKTV(:,2)))>0 & (cpt_y+max(NetKTV(:,2)))<=Ny
                                        if (cpt_z+min(NetKTV(:,3)))>0 & (cpt_z+max(NetKTV(:,3)))<=Nz
                                            if ACS(cpt_x,cpt_y,cpt_z)~=0
                                                list_pts(cpt,:)=[cpt_x,cpt_y,cpt_z];
                                                cpt=cpt+1;
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
        end
        function [k_composite_gpu]=Create_KComposite_GPU(kspace,Ry,Rz,shift_calc)
            if nargin < 4
                shift_calc=[];
            end
                kspace_gpu=gpuArray(single(kspace));
                k_composite_gpu=gpuArray(single(zeros(size(kspace_gpu(:,:,:,:,1,1)))));
                for cpt_v=1:1:min(size(kspace_gpu,5),Ry)
                    for cpt_t=1:1:min(size(kspace_gpu,6),Rz)
                         if isempty(shift_calc)
                                 [idY idZ]=find(squeeze(k_compose(end/2,:,:,1,cptV,cptCar)));
                                initialShiftY=idY(1);
                                initialShiftZ=idZ(1);
                            else
                                initialShiftY=shift_calc(cpt_v,cpt_t,1);
                                initialShiftZ=shift_calc(cpt_v,cpt_t,2);
                         end
                         k_composite_gpu(:,initialShiftY:Ry:end,initialShiftZ:Rz:end,:)=kspace_gpu(:,initialShiftY:Ry:end,initialShiftZ:Rz:end,:,cpt_v,cpt_t);
                    end
                end
        
        end

          function [k_R,ACS]=Undersample_kspace(kspace,Ry,Rz,acs_size,interleaving)
            % Undersample the data with or without shift
            [Nx Ny Nz Nc Nvenc Ncardiac]=size(kspace);
            k_R=zeros(size(kspace)); 
            ACS=zeros([Nx acs_size(1)*2 acs_size(2)*2 Nc Nvenc Ncardiac]);
            for cptY=1:Ry:(Ny)
                for cptZ=1:Rz:(Nz)
                    for cptV=1:1:Nvenc
                        if interleaving==1
                            shiftY=mod(cptV-1,Ry);
                            shiftY2=shiftY;
                        elseif interleaving==2
                            shiftY=mod(cptV-1,Ry);
                            shiftY2=0;
                        else
                            shiftY=0;
                            shiftY2=shiftY;
                        end
                        for cptCar=1:1:Ncardiac
                            if interleaving==1
                                shiftZ=mod(cptCar-1,Rz);
                                shiftZ2=shiftZ;
                            elseif interleaving==2
                                shiftZ=mod(cptCar-1,Rz);
                                shiftZ2=0;
                            else
                                shiftZ=0;
                                shiftZ2=shiftZ;
                            end
                           
                            if cptY+shiftY<=Ny && cptZ+shiftZ<=Nz
                               k_R(:,cptY+shiftY,cptZ+shiftZ,:,cptV,cptCar)=kspace(:,cptY+shiftY2,cptZ+shiftZ2,:,cptV,cptCar);
                            end

                        end
                    end
                end
            end

            % Add the center lines as ACS lines
            k_R(:,end/2-acs_size(1)+1:end/2+acs_size(1)+1,end/2-acs_size(2)+1:end/2+acs_size(2)+1,:,:,:)=kspace(:,end/2-acs_size(1)+1:end/2+acs_size(1)+1,end/2-acs_size(2)+1:end/2+acs_size(2)+1,:,:,:);
            ACS=kspace(:,end/2-acs_size(1)+1:end/2+acs_size(1)+1,end/2-acs_size(2)+1:end/2+acs_size(2)+1,:,:,:);
            
          end

        function NetKTV=getNetKTV(Ry,Rz,Nc,Mode)
            NetKTV=[];
            cK=0;
            for cZ=0:1:Rz-1
                for cY=0:1:Ry-1
                    % create the network of point corresponding to ktv-GRAPPA
                    NetX=GRAPPA_5D_ktv.NetX;
                    if Mode==0
                        NetT=GRAPPA_5D_ktv.NetT0;
                        NetVenc=GRAPPA_5D_ktv.NetVenc0;
                    elseif Mode==1
                        NetT=GRAPPA_5D_ktv.NetT(1:min(length(GRAPPA_5D_ktv.NetT),Rz));
                        NetVenc=GRAPPA_5D_ktv.NetVenc0;
                    else
                        NetT=GRAPPA_5D_ktv.NetT(1:min(length(GRAPPA_5D_ktv.NetT),Rz));
                        NetVenc=GRAPPA_5D_ktv.NetVenc(1:min(length(GRAPPA_5D_ktv.NetVenc),Ry));
                    end
                    if cK>0
                        NetYZ(:,:,cK)=[0-cY 0-cZ;0-cY Rz-cZ;Ry-cY 0-cZ; Ry-cY Rz-cZ];
                        cpt=1;
                        for cpt_x=1:1:length(NetX)
                            for cpt_yz=1:1:size(NetYZ,1)
                                for cpt_v=1:1:length(NetVenc)
                                    for cpt_t=1:1:length(NetT)
                                        for cC=1:Nc 
                                              NetKTV(cpt,1,cK)= NetX(cpt_x);
                                              NetKTV(cpt,2,cK)= NetYZ(cpt_yz,1,cK)+NetVenc(cpt_v);
                                              NetKTV(cpt,3,cK)= NetYZ(cpt_yz,2,cK)+NetT(cpt_t);
                                              NetKTV(cpt,4,cK)=cC;
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
        end
        
        function coil=ESPIRIT_KM(Raw)
        % Raw [nx ny nc nz] coil images -> coil [nx ny nc nz] sensitivity maps
        %
        % Vectorised version of the original per-voxel loop. For each voxel the
        % dominant eigenvector of the local coil covariance over a (2w+1)^3
        % window (truncated at the borders) is computed. A window truncated at
        % the borders gives the same sum as a zero-padded box filter, so all
        % covariances are obtained with separable convn calls, processed in
        % x-slabs to bound memory. Results match the original to rounding error.
            data = permute(Raw,[1 2 4 3]);                      % [Nx Ny Nz Nc]
            [Nx,Ny,Nz,Nc] = size(data);
            w = 5;
            k = 2*w+1;
            box = @(A) convn(convn(convn(A,ones(k,1),'same'),ones(1,k),'same'),ones(1,1,k),'same');
            usePageEig = exist('pageeig') > 0;                  % R2023a+

            S = complex(zeros(Nx,Ny,Nz,Nc));
            M = zeros(Nx,Ny,Nz);
            slab = max(1,floor(512e6/(Ny*Nz*Nc^2*16)));         % ~512 MB of covariances per slab
            for x0 = 1:slab:Nx
                x1 = min(Nx,x0+slab-1);
                h0 = max(1,x0-w);  h1 = min(Nx,x1+w);           % slab + halo
                d  = double(data(h0:h1,:,:,:));
                R  = complex(zeros(h1-h0+1,Ny,Nz,Nc,Nc));
                for a = 1:Nc
                    R(:,:,:,a,a) = real(box(abs(d(:,:,:,a)).^2));
                    for b = a+1:Nc
                        s = box(d(:,:,:,a).*conj(d(:,:,:,b)));  % = conj(kernel'*kernel)(a,b)
                        R(:,:,:,a,b) = s;
                        R(:,:,:,b,a) = conj(s);
                    end
                end
                ns = x1-x0+1;
                R  = R(x0-h0+1:x0-h0+ns,:,:,:,:);
                np = ns*Ny*Nz;
                R  = permute(reshape(R,np,Nc,Nc),[2 3 1]);      % [Nc Nc np]

                if usePageEig
                    [V,D] = pageeig(R);
                    D = real(reshape(D,Nc*Nc,np));
                    [dmax,imax] = max(D(1:Nc+1:end,:),[],1);    % largest eigenvalue per voxel
                    V = reshape(V,Nc,Nc*np);
                    v = V(:,(0:np-1)*Nc+imax);                  % [Nc np]
                else                                            % older releases
                    v = complex(zeros(Nc,np));
                    dmax = zeros(1,np);
                    for p = 1:np
                        [Vp,Dp] = eig(R(:,:,p));
                        [dmax(p),i] = max(real(diag(Dp)));
                        v(:,p) = Vp(:,i);
                    end
                end
                v = v.*exp(-1j*angle(v(1,:)));                  % same phase reference
                S(x0:x1,:,:,:) = reshape(v.',ns,Ny,Nz,Nc);
                M(x0:x1,:,:)   = reshape(sqrt(max(dmax,0)),ns,Ny,Nz);
            end
            coil = permute(squeeze(S.*(M>0.01*max(abs(M(:))))),[1 2 4 3]);
        end
        
        function [mask_YZ, listKvc]=Mask_discovery(k_compose,Ry,Rz)
                mask_YZ=zeros(size(squeeze(k_compose(:,:,:,1,:,:))));
                [Nx, Ny, Nz, Nc, Nvenc, Ncardiac] = size(k_compose);
                k_mask=squeeze(max(k_compose,[],[1 4])); % 4D : Ky Kz Kvenc Kcardiac
                k_mask(k_mask~=0)=1;
                listKvc=zeros(Nvenc,Ncardiac,2);

                for cpt_kv=1:1:Nvenc
                    for cpt_kc=1:1:Ncardiac
                        tmp_0=0;
                        for cpt_v=1:1:Ry
                            for cpt_t=1:1:Rz
                                tmp_mask=zeros(Ny,Nz);
                                tmp_mask(cpt_v:Ry:end,cpt_t:Rz:end)=1;
                                tmp_mask=tmp_mask.*k_mask(:,:,cpt_kv,cpt_kc);
                                tmp=sum(tmp_mask(:));
                                if tmp>tmp_0
                                    tmp_0=tmp;
                                    listKvc(cpt_kv,cpt_kc,:)=[cpt_v,cpt_t];
                                end
                            end
                        end
                    end
                end
                cK=0;
                for cZ=0:1:Rz-1
                    for cY=0:1:Ry-1
                        for cptV=1:1:Nvenc
                            for cptCar=1:1:Ncardiac
                                mask_YZ(:,listKvc(cptV,cptCar,1)+cY:Ry:end,listKvc(cptV,cptCar,2)+cZ:Rz:end,cptV,cptCar)=cK; % The shift is based on the regular pattern. 
                                mask_YZ(:,listKvc(cptV,cptCar,1)+cY:-Ry:1,listKvc(cptV,cptCar,2)+cZ:-Rz:1,cptV,cptCar)=cK; % The shift is based on the regular pattern. 
                                mask_YZ(:,listKvc(cptV,cptCar,1)+cY:-Ry:1,listKvc(cptV,cptCar,2)+cZ:Rz:end,cptV,cptCar)=cK; % The shift is based on the regular pattern.
                                mask_YZ(:,listKvc(cptV,cptCar,1)+cY:Ry:end,listKvc(cptV,cptCar,2)+cZ:-Rz:1,cptV,cptCar)=cK; % The shift is based on the regular pattern.
                            end
                        end
                        cK=cK+1;
                    end
                end
        end
            function [k_composite_gpu]=Create_Composite_GPU(kspace,list_mask,Ry,Rz,current_V,current_Car,no_shift)

            if nargin<7
                no_shift=0;
            end
            % This logic works for 4 points in Venc and 3 points in Time
            % has to be adapted for other data format
            k_composite_gpu=gpuArray(single(zeros(size(kspace(:,:,:,:,1,1)))));
            Nvenc=size(kspace,5);
            Ncar=size(kspace,6);
            Mvenc=min(GRAPPA_5D_ktv.MaxVencPts,Ry);
            Mcar=min(GRAPPA_5D_ktv.MaxTimePts,Rz);

            %%% check up front that enough distinct sampling shifts exist
            shiftsY=reshape(list_mask(:,1,1),1,[]);   % ky shift of each venc frame (cardiac phase 1)
            shiftsZ=reshape(list_mask(1,:,2),1,[]);   % kz shift of each cardiac frame (venc 1)
            if numel(unique(shiftsY))<Mvenc || numel(unique(shiftsZ))<Mcar
                error('GRAPPA_5D_ktv:CompositeShifts', ...
                    ['Create_Composite_GPU needs %d distinct ky shifts across venc frames and %d distinct kz shifts ' ...
                     'across cardiac frames, but found ky shifts %s and kz shifts %s. ' ...
                     'k-t-v GRAPPA requires sampling interleaved across venc (ky) and time (kz).'], ...
                    Mvenc,Mcar,mat2str(shiftsY),mat2str(shiftsZ));
            end

            list_venc=[];
            list_venc(1)=current_V;
            cpt_tmp=1;
            cpt_tmp_c=current_V;
            n_iter=0;                                   
            while (cpt_tmp<Mvenc)
                n_iter=n_iter+1;                  
                if n_iter>Nvenc
                    error('GRAPPA_5D_ktv:CompositeVenc', ...
                        'Could not find %d distinct ky shifts across venc frames (found %s).', ...
                        Mvenc,mat2str(reshape(list_mask(list_venc,1,1),1,[])));
                end
                cpt_tmp_c=cpt_tmp_c+1;
                if cpt_tmp_c>Nvenc
                    cpt_tmp_c=1;
                end
                if (~any(list_mask(cpt_tmp_c,1,1)==list_mask(list_venc,1,1)))
                     cpt_tmp=cpt_tmp+1;
                     list_venc(cpt_tmp)=cpt_tmp_c;
                end
            end

            list_car=[];
            list_car(1)=current_Car;
            cpt_tmp=1;
            if current_Car~=1
                cpt_tmp_c=current_Car+Rz-1;
            else
                cpt_tmp_c=current_Car;
            end   % (for current_Car==1, cpt_tmp_c continues from the venc loop, as before)
            n_iter=0;                                 
            while (cpt_tmp<Mcar)
                n_iter=n_iter+1;                       
                if n_iter>2*Ncar+3
                    error('GRAPPA_5D_ktv:CompositeTime', ...
                        'Could not find %d distinct kz shifts among the cardiac frames reachable from phase %d (found %s).', ...
                        Mcar,current_Car,mat2str(reshape(list_mask(1,list_car,2),1,[])));
                end
                cpt_tmp_c=cpt_tmp_c-1;
                if cpt_tmp_c<1
                    cpt_tmp_c=Ncar-cpt_tmp_c;
                end
                % if current_Car==1&&cpt_tmp_c==Ncar-2 % added rule so we don't got too backward for the first cardiac phase.
                %         cpt_tmp_c=current_Car+2;
                % end
                if cpt_tmp_c>Ncar
                    cpt_tmp_c=1;
                end
                if (~any(list_mask(1,cpt_tmp_c,2)==list_mask(1,list_car,2)))
                     cpt_tmp=cpt_tmp+1;
                     list_car(cpt_tmp)=cpt_tmp_c;
                end
            end
            disp(['composite using points venc ' num2str(list_venc) ' and points time ' num2str(list_car)])
            for cpt_v=1:1:Mvenc
                for cpt_t=1:1:Mcar
                     initialShiftY=list_mask(list_venc(cpt_v),list_car(cpt_t),1);
                     initialShiftZ=list_mask(list_venc(cpt_v),list_car(cpt_t),2);
                     initialShiftY2=initialShiftY;
                     initialShiftZ2=initialShiftZ;
                     if no_shift~=0
                        initialShiftY2=1;
                        initialShiftZ2=1;
                     end
                     k_composite_gpu(:,initialShiftY:Ry:end,initialShiftZ:Rz:end,:)=kspace(:,initialShiftY2:Ry:end-(initialShiftY-initialShiftY2),initialShiftZ2:Rz:end-(initialShiftZ-initialShiftZ2),:,list_venc(cpt_v),list_car(cpt_t));
                end
            end
        end
        function [k_composite_gpu,kspace_gpu]=Create_Composite_GPU_T(kspace,list_mask,Ry,Rz,current_V,current_Car,no_shift)
            
            if nargin<7
                no_shift=0;
            end
            % This logic works for 4 points in Venc and 3 points in Time 
            % has to be adapted for other data format
           
            %%
            Mvenc_unic=max(list_mask(:,:,1),[],'all');
            Mtime_unic=max(list_mask(:,:,2),[],'all');
            for cpt_v=1:1:Mvenc_unic
                for cpt_t=1:1:Mtime_unic
                    Idx=find( list_mask(:,:,1)==cpt_v & list_mask(:,:,2)==cpt_t );
                    [Idx_v, Idx_t] = ind2sub(size(list_mask(:,:,1)), Idx); 
                    kspace_ave(:,:,:,:,cpt_v,cpt_t)=mean(kspace(:,:,:,:,Idx_v,Idx_t),[5, 6]);
                end
            end
            list_mask2=list_mask(1:Mvenc_unic,1:Mtime_unic,:);
            [k_composite_gpu]=GRAPPA_5D_ktv.Create_Composite_GPU(kspace_ave,list_mask2,Ry,Rz,list_mask(current_V,current_Car,1),list_mask(current_V,current_Car,2),no_shift);
            kspace_gpu=gpuArray(kspace_ave(:,:,:,:,list_mask(current_V,current_Car,1),list_mask(current_V,current_Car,2)));
        end
    end
end

