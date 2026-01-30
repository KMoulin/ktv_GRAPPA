

function Recon_n_Train_5D_2026_2_compile_mex()
    % Compilation script for GRAPPA 5D CUDA MEX function
    %
    % This script compiles the CUDA MEX implementation with optimized settings
    setenv("NVCC_APPEND_FLAGS", '--allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH');
    fprintf('=== Compiling GRAPPA 5D CUDA MEX ===\n');
    
    % Check CUDA availability
    if ~parallel.gpu.GPUDevice.isAvailable
        error('No CUDA-capable GPU detected');
    end
    
    % Get GPU info
    gpu = gpuDevice;
    fprintf('GPU: %s (Compute Capability %s)\n', gpu.Name, gpu.ComputeCapability);
    
    % Determine architecture flag
    cc = str2double(gpu.ComputeCapability);
    if cc >= 8.9
        arch = 'sm_89';  % RTX 40xx
    elseif cc >= 8.6
        arch = 'sm_86';  % RTX 30xx
    elseif cc >= 8.0
        arch = 'sm_80';  % A100
    elseif cc >= 7.5
        arch = 'sm_75';  % RTX 20xx
    elseif cc >= 7.0
        arch = 'sm_70';  % V100
    elseif cc >= 6.0
        arch = 'sm_60';  % P100
    else
        arch = 'sm_35';  % Minimum
        warning('Old GPU architecture detected. Performance may be limited.');
    end
    
    fprintf('Using architecture: %s\n', arch);
    
    % Source file
    source_file = 'Recon_n_Train_5D_2026_2.cu';
    
    if ~exist(source_file, 'file')
        error('Source file %s not found', source_file);
    end
    
    % Compilation options
    nvcc_flags = sprintf('-O3 -use_fast_math -arch=%s', arch);
    
    % Additional flags for debugging (comment out for release)
    % nvcc_flags = [nvcc_flags ' -G -lineinfo'];
    
    % Compiler command
    fprintf('\nCompiling with flags: %s\n', nvcc_flags);
    fprintf('This may take a minute...\n\n');
    %'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64';
    try
        mexcuda('-R2018a', source_file, ...
                '-lcublas', '-lcusolver', ...
                '-LC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64',...
                '-IC:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include', ...
                ['NVCCFLAGS=' nvcc_flags]);
        
        fprintf('\n=== Compilation Successful! ===\n');
        fprintf('MEX file: %s\n', [source_file(1:end-3) '.' mexext]);
        
        % Test if it loads
        fprintf('\nTesting MEX function load...\n');
        if exist('Recon_n_Train_5D_2026_2_complete', 'file')
            fprintf('✓ MEX function is ready to use\n');
        else
            warning('MEX file compiled but not found in path');
        end
        
    catch ME
        fprintf('\n=== Compilation Failed ===\n');
        fprintf('Error: %s\n', ME.message);
        
        % Provide troubleshooting hints
        fprintf('\nTroubleshooting:\n');
        fprintf('1. Ensure CUDA toolkit is installed and in PATH\n');
        fprintf('2. Try: mex -setup C++\n');
        fprintf('3. Check GPU compute capability is >= 3.5\n');
        fprintf('4. Verify cuBLAS and cuSOLVER are available\n');
        
        rethrow(ME);
    end
    
    fprintf('\n=== Usage Example ===\n');
    fprintf('k_recon = Recon_n_Train_5D_2026_2(k_gpu, ACS_gpu, mask_gpu, NetKTV, Ry, Rz);\n\n');
end