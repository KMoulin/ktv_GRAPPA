/*
 * MEX CUDA implementation of GRAPPA 5D reconstruction
 * Function signature: k_recon = Recon_n_Train_5D_2026_2(k_composite_gpu, k_R_gpu, 
 *                     ACS_composite_gpu, ACS_gpu, mask_R, NetKTV, Ry, Rz)
 * 
 * Compile with: mexcuda -R2018a Recon_n_Train_5D_2026_2.cu -lcublas -lcusolver
 */

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "mat.h"

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            mexErrMsgIdAndTxt("GRAPPA:cudaError", \
                "CUDA error at %s:%d: %s", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            mexErrMsgIdAndTxt("GRAPPA:cublasError", \
                "cuBLAS error at %s:%d: %d", __FILE__, __LINE__, status); \
        } \
    } while(0)

#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            mexErrMsgIdAndTxt("GRAPPA:cusolverError", \
                "cuSOLVER error at %s:%d: %d", __FILE__, __LINE__, status); \
        } \
    } while(0)

// ============================================================================
// GPU KERNEL: Extract patches from composite data
// ============================================================================
__global__ void extractPatchesKernel_GPU(
    const cuFloatComplex* vect_mat,
    const int* list_points,
    const int* NetKTV,
    cuFloatComplex* Sn,
    int num_points,
    int dim_patch,
    int patch_size,
    int ck,
    int Nx, int Ny, int Nz, int Nc)
{
    // Each thread processes one point
    int pts_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pts_idx >= num_points) return;
    
    // Extract all patches for this point
    for (int patch_idx = 0; patch_idx < patch_size; patch_idx++)
    {
        int offset_base = ck * patch_size * dim_patch;
        
        int x = list_points[pts_idx*3 + 0] + NetKTV[patch_idx + patch_size*0 + offset_base];
        int y = list_points[pts_idx*3 + 1] + NetKTV[patch_idx + patch_size*1 + offset_base];
        int z = list_points[pts_idx*3 + 2] + NetKTV[patch_idx + patch_size*2 + offset_base];
        int c = 							 NetKTV[patch_idx + patch_size*3 + offset_base]-1; // Change of base matlab/Cuda  here
        
        // Bounds checking
        if (x >= 0 && x < Nx && y >= 0 && y < Ny && z >= 0 && z < Nz && c >= 0 && c < Nc)
        {
            // MATLAB column-major indexing
            int src_idx = x + y*Nx + z*Nx*Ny + c*Nx*Ny*Nz;
            int dst_idx = patch_idx + pts_idx*patch_size;
            Sn[dst_idx] = vect_mat[src_idx];
        }
    }
}

// Host wrapper for extractPatchesKernel
void extractPatchesKernel(
    const cuFloatComplex* vect_mat,
    const int* list_points,
    const int* NetKTV,
    cuFloatComplex* Sn,
    int num_points,
    int dim_patch,
    int patch_size,
    int ck,
    int Nx, int Ny, int Nz, int Nc)
{
    if (num_points == 0) return;
    
    // Configure kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;
    
    mexPrintf("  Launching extractPatches: %d blocks x %d threads = %d threads (for %d points)\n",
              blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock, num_points);
    // Validate pointers
    if (vect_mat == NULL) mexErrMsgIdAndTxt("GRAPPA:nullPointer", "vect_mat is NULL");
    if (list_points == NULL)mexErrMsgIdAndTxt("GRAPPA:nullPointer", "list_points is NULL");
    if (NetKTV == NULL) mexErrMsgIdAndTxt("GRAPPA:nullPointer", "NetKTV is NULL");
    if (Sn == NULL) mexErrMsgIdAndTxt("GRAPPA:nullPointer", "Sn is NULL");
	mexPrintf("    num_points  = %d\n", num_points);
    mexPrintf("    dim_patch   = %d\n", dim_patch);
    mexPrintf("    patch_size  = %d\n", patch_size);
    mexPrintf("    ck          = %d\n", ck);
    mexPrintf("    Nx=%d, Ny=%d, Nz=%d, Nc=%d\n", Nx, Ny, Nz, Nc);
    // Launch kernel
    extractPatchesKernel_GPU<<<blocksPerGrid, threadsPerBlock>>>(vect_mat, list_points, NetKTV, Sn, num_points, dim_patch, patch_size, ck, Nx, Ny, Nz, Nc);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// GPU KERNEL: Extract ACS target values
// ============================================================================
__global__ void extractACSKernel_GPU(
    const cuFloatComplex* vect_mat,
    const int* list_points,
    cuFloatComplex* Xn,
    int num_points,
    int Nx, int Ny, int Nz, int Nc)
{
    // Each thread processes one point
    int pts_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pts_idx >= num_points) return;
    
    int x = list_points[pts_idx*3 + 0];
    int y = list_points[pts_idx*3 + 1];
    int z = list_points[pts_idx*3 + 2];
    
    // Bounds checking
    if (x >= 0 && x < Nx && y >= 0 && y < Ny && z >= 0 && z < Nz)
    {
        // Extract all coils for this point
        for (int coil_idx = 0; coil_idx < Nc; coil_idx++)
        {
            int src_idx = x + y*Nx + z*Nx*Ny + coil_idx*Nx*Ny*Nz;
            int dst_idx = coil_idx + pts_idx*Nc;
            Xn[dst_idx] = vect_mat[src_idx];
        }
    }
}

// Host wrapper for extractACSKernel
void extractACSKernel(
    const cuFloatComplex* vect_mat,
    const int* list_points,
    cuFloatComplex* Xn,
    int num_points,
    int Nx, int Ny, int Nz, int Nc)
{
    if (num_points == 0) return;
    
    // Configure kernel launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;
    
    mexPrintf("  Launching extractACS: %d blocks x %d threads = %d threads (for %d points)\n",
              blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock, num_points);
    
    // Launch kernel
    extractACSKernel_GPU<<<blocksPerGrid, threadsPerBlock>>>(
        vect_mat, list_points, Xn, num_points, Nx, Ny, Nz, Nc);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// GPU KERNEL: Assign reconstructed values to k-space
// ============================================================================
__global__ void assignReconValuesKernel_GPU(
    cuFloatComplex* k_recon,
    const cuFloatComplex* X_recon,
    const int* list_points,
    int num_points,
    int Nx, int Ny, int Nz, int Nc)
{
    // 2D grid: one dimension for points, one for coils
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int coil_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (point_idx >= num_points || coil_idx >= Nc) return;
    
    // Get coordinates
    int x = list_points[point_idx * 3 + 0];
    int y = list_points[point_idx * 3 + 1];
    int z = list_points[point_idx * 3 + 2];
    
    // Bounds checking
    if (x >= 0 && x < Nx && y >= 0 && y < Ny && z >= 0 && z < Nz)
    {
        // Calculate linear index in k_recon
        int lin_idx = x + y * Nx + z * Nx * Ny + coil_idx * Nx * Ny * Nz;
        
        // Assign value (X is Nc x num_points, column-major)
        k_recon[lin_idx] = X_recon[coil_idx + point_idx * Nc];
    }
}

// Host wrapper for assignReconValuesKernel
void assignReconValuesKernel(
    cuFloatComplex* k_recon,
    const cuFloatComplex* X_recon,
    const int* list_points,
    int num_points,
    int Nx, int Ny, int Nz, int Nc)
{
    if (num_points == 0) return;
    
    // Configure 2D kernel launch
    dim3 threadsPerBlock(16, 16);  // 16x16 = 256 threads per block
    dim3 blocksPerGrid(
        (num_points + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (Nc + threadsPerBlock.y - 1) / threadsPerBlock.y
    );
    
    mexPrintf("  Launching assignRecon: (%d,%d) blocks x (%d,%d) threads\n",
              blocksPerGrid.x, blocksPerGrid.y, 
              threadsPerBlock.x, threadsPerBlock.y);
    
    // Launch kernel
    assignReconValuesKernel_GPU<<<blocksPerGrid, threadsPerBlock>>>(
        k_recon, X_recon, list_points, num_points, Nx, Ny, Nz, Nc);
    
    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// HOST FUNCTION: Find reconstruction points where mask == cK
// ============================================================================
int findReconPointsKernel(
    const double* mask,
    int** d_list_pts,
    int target_value,
    int Nx, int Ny, int Nz)
{
    int total = Nx * Ny * Nz;
    int pos = 0;
    thrust::host_vector<int> h_list;
    int x, y, z, remainder;
    
    // CPU loop to find matching points
    for (int i = 0; i < total; i++)
    {    
        if (mask[i] == target_value) 
        {
            // Convert linear index to 3D coordinates (column-major)
            x = i % Nx;
            remainder = i / Nx;
            y = remainder % Ny;
            z = remainder / Ny;
            
            h_list.push_back(x);
            h_list.push_back(y);
            h_list.push_back(z);
            
            pos++;
        }
    }
    
    if (pos > 0)
    {
        CUDA_CHECK(cudaMalloc(d_list_pts, h_list.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(*d_list_pts, thrust::raw_pointer_cast(h_list.data()),
                             h_list.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    return pos;
}

// ============================================================================
// HOST FUNCTION: Get list of training examples
// ============================================================================
int getListOfExamples(
    int Nx, int Ny, int Nz,
    const int* NetKTV,
    int patch_size,
    int dim_patch,
    int ck,
    int** d_list_pts)
{

    // Find min and max offsets in NetKTV
    int min_x = 0, max_x = 0;
    int min_y = 0, max_y = 0;
    int min_z = 0, max_z = 0;
    int num_examples = 0;
    
    for (int i = 0; i < patch_size; i++) 
    {
        int ox = NetKTV[i + 0*patch_size + ck*patch_size*dim_patch];
        int oy = NetKTV[i + 1*patch_size + ck*patch_size*dim_patch];
        int oz = NetKTV[i + 2*patch_size + ck*patch_size*dim_patch];
        
        if (i == 0) 
        {
            min_x = max_x = ox;
            min_y = max_y = oy;
            min_z = max_z = oz;
        } 
        else 
        {
            min_x = min(min_x, ox);
            max_x = max(max_x, ox);
            min_y = min(min_y, oy);
            max_y = max(max_y, oy);
            min_z = min(min_z, oz);
            max_z = max(max_z, oz);
        }
    }
    
    mexPrintf("  Patch bounds: X[%d,%d] Y[%d,%d] Z[%d,%d]\n", min_x, max_x, min_y, max_y, min_z, max_z);
    
    // Create list of valid points
    thrust::host_vector<int> h_list;

    for (int x = 0; x < Nx; x++) 
    {
        for (int y = 0; y < Ny; y++) 
        {
            for (int z = 0; z < Nz; z++) 
            {
                if ((x + min_x) >= 0 && (x + max_x) < Nx &&
                    (y + min_y) >= 0 && (y + max_y) < Ny && 
                    (z + min_z) >= 0 && (z + max_z) < Nz) 
                {
                    h_list.push_back(x);
                    h_list.push_back(y);
                    h_list.push_back(z);
					num_examples++;
                }
            }
        }
    }
    
   // num_examples = h_list.size() / 3;
    
    if (num_examples > 0) 
    {
        CUDA_CHECK(cudaMalloc(d_list_pts, h_list.size() * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(*d_list_pts, thrust::raw_pointer_cast(h_list.data()),
                             h_list.size() * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    return num_examples;
}
// ============================================================================
// Solve GRAPPA weights: W = (X * S^H) / (S * S^H)
// ============================================================================
/*
void solveGRAPPAWeights(
    cublasHandle_t cublas_handle,
    cusolverDnHandle_t cusolver_handle,
    const cuFloatComplex* d_Xn,  // Nc x num_examples
    const cuFloatComplex* d_Sn,  // patch_size x num_examples
    cuFloatComplex* d_Wn,        // Nc x patch_size (output)
    int Nc,
    int patch_size,
    int num_examples)
{
    cuFloatComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
    cuFloatComplex beta = make_cuFloatComplex(0.0f, 0.0f);
    
    // Compute S * S^H (patch_size x patch_size)
    cuFloatComplex* d_SSH;
    CUDA_CHECK(cudaMalloc(&d_SSH, patch_size * patch_size * sizeof(cuFloatComplex)));
   
   // Make GEMM deterministic-ish / no TF32 tricks
	CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_PEDANTIC_MATH));
	
    CUBLAS_CHECK(cublasCgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_C,
        patch_size, patch_size, num_examples,
        &alpha, d_Sn, patch_size, d_Sn, patch_size,
        &beta, d_SSH, patch_size));
    
    // Compute X * S^H (Nc x patch_size)
    cuFloatComplex* d_XSH;
    CUDA_CHECK(cudaMalloc(&d_XSH, Nc * patch_size * sizeof(cuFloatComplex)));
    
    CUBLAS_CHECK(cublasCgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_C,
        Nc, patch_size, num_examples,
        &alpha, d_Xn, Nc, d_Sn, patch_size,
        &beta, d_XSH, Nc));
    
    // Solve (S*S^H) * W^T = (X*S^H)^T for W using LU decomposition
    int* d_info;
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnCgetrf_bufferSize(cusolver_handle, 
        patch_size, patch_size, d_SSH, patch_size, &lwork));
    
    cuFloatComplex* d_work;
    CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(cuFloatComplex)));
    
    int* d_ipiv;
    CUDA_CHECK(cudaMalloc(&d_ipiv, patch_size * sizeof(int)));
    
    // LU factorization of S*S^H
    CUSOLVER_CHECK(cusolverDnCgetrf(cusolver_handle, patch_size, patch_size, 
        d_SSH, patch_size, d_work, d_ipiv, d_info));
    
    // Transpose X*S^H for solving
    cuFloatComplex* d_XSH_T;
    CUDA_CHECK(cudaMalloc(&d_XSH_T, Nc * patch_size * sizeof(cuFloatComplex)));
    
    CUBLAS_CHECK(cublasCgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        patch_size, Nc, &alpha, d_XSH, Nc, &beta, d_XSH_T, patch_size,
        d_XSH_T, patch_size));
    
    // Solve for W^T
    CUSOLVER_CHECK(cusolverDnCgetrs(cusolver_handle, CUBLAS_OP_N, 
        patch_size, Nc, d_SSH, patch_size, d_ipiv, d_XSH_T, patch_size, d_info));
    
    // Transpose back to get W (Nc x patch_size)
    CUBLAS_CHECK(cublasCgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        Nc, patch_size, &alpha, d_XSH_T, patch_size, &beta, d_Wn, Nc,
        d_Wn, Nc));
    
    // Cleanup
    cudaFree(d_SSH);
    cudaFree(d_XSH);
    cudaFree(d_XSH_T);
    cudaFree(d_work);
    cudaFree(d_ipiv);
    cudaFree(d_info);
}
*/

// ============================================================================
// Solve GRAPPA weights via least squares (QR decomposition)
// Solves: S^T * W^T ≈ X^T  =>  W^T ≈ (S^T)^+ * X^T
// ============================================================================

void solveGRAPPAWeights(
    cublasHandle_t cublas_handle,
    cusolverDnHandle_t cusolver_handle,
    const cuFloatComplex* d_Xn,  // Nc x num_examples
    const cuFloatComplex* d_Sn,  // patch_size x num_examples
    cuFloatComplex* d_Wn,        // Nc x patch_size (output)
    int Nc,
    int patch_size,
    int num_examples)
{
    const cuFloatComplex alpha = make_cuFloatComplex(1.f, 0.f);
    const cuFloatComplex beta  = make_cuFloatComplex(0.f, 0.f);
    
    int m = num_examples;      // Number of equations
    int n = patch_size;        // Number of unknowns
    int nrhs = Nc;             // Number of right-hand sides
    
    // ========================================================================
    // Step 1: Transpose inputs
    // A = S^T  (m x n)
    // B = X^T  (m x nrhs)
    // ========================================================================
    cuFloatComplex* d_A;
    cudaMalloc(&d_A, (size_t)m * n * sizeof(cuFloatComplex));
    cublasCgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        m, n, &alpha, d_Sn, patch_size, &beta, d_A, m, d_A, m);
    
    cuFloatComplex* d_B;
    cudaMalloc(&d_B, (size_t)m * nrhs * sizeof(cuFloatComplex));
    cublasCgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        m, nrhs, &alpha, d_Xn, Nc, &beta, d_B, m, d_B, m);
    
    // ========================================================================
    // Step 2: QR factorization of A = Q * R
    // ========================================================================
    cuFloatComplex* d_tau;
    cudaMalloc(&d_tau, n * sizeof(cuFloatComplex));
    
    int* d_info;
    cudaMalloc(&d_info, sizeof(int));
    
    // Query workspace size
    int lwork_geqrf = 0;
    cusolverDnCgeqrf_bufferSize(cusolver_handle, m, n, d_A, m, &lwork_geqrf);
    
    cuFloatComplex* d_work;
    cudaMalloc(&d_work, lwork_geqrf * sizeof(cuFloatComplex));
    
    // Compute QR factorization
    cusolverDnCgeqrf(cusolver_handle, m, n, d_A, m, d_tau, d_work, lwork_geqrf, d_info);
    
    // ========================================================================
    // Step 3: Compute Q^H * B
    // ========================================================================
    int lwork_unmqr = 0;
    cusolverDnCunmqr_bufferSize(cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_C,
        m, nrhs, n, d_A, m, d_tau, d_B, m, &lwork_unmqr);
    
    // Reallocate workspace if needed
    if (lwork_unmqr > lwork_geqrf) {
        cudaFree(d_work);
        cudaMalloc(&d_work, lwork_unmqr * sizeof(cuFloatComplex));
    }
    
    // Apply Q^H to B: B := Q^H * B
    cusolverDnCunmqr(cusolver_handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_C,
        m, nrhs, n, d_A, m, d_tau, d_B, m, d_work, lwork_unmqr, d_info);
    
    // ========================================================================
    // Step 4: Solve R * X = (Q^H * B) for each right-hand side
    // R is stored in upper triangle of d_A
    // ========================================================================
    for (int i = 0; i < nrhs; i++) {
        cuFloatComplex* b_col = d_B + (size_t)i * m;
        
        // Solve R * x = b  (triangular solve)
        cublasCtrsm(cublas_handle,
            CUBLAS_SIDE_LEFT,
            CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT,
            n, 1,
            &alpha,
            d_A, m,
            b_col, m);
    }
    
    // ========================================================================
    // Step 5: Transpose result back: W = B(1:n, :)^T
    // ========================================================================
    cublasCgeam(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
        Nc, patch_size, &alpha, d_B, m, &beta, d_Wn, Nc, d_Wn, Nc);
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_tau);
    cudaFree(d_work);
    cudaFree(d_info);
}

// ============================================================================
// DEBUG: Write matrix to MATLAB workspace (simpler, no file I/O)
// ============================================================================
void writeToWorkspace(const char* varName, const mwSize *dims, cuFloatComplex* d_data)
{
    size_t numElements = dims[0] * dims[1];
    
    mexPrintf("Writing %s [%zu x %zu] to workspace...\n", varName, dims[0], dims[1]);
    
    // Allocate and copy
    cuFloatComplex *h_temp = (cuFloatComplex*)malloc(numElements * sizeof(cuFloatComplex));
    CUDA_CHECK(cudaMemcpy(h_temp, d_data, numElements * sizeof(cuFloatComplex), 
                          cudaMemcpyDeviceToHost));
    
    // Create mxArray
    mxArray *cpuArray = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxCOMPLEX);
    mxComplexSingle *cpuData = mxGetComplexSingles(cpuArray);
    
    // Convert
    for (size_t i = 0; i < numElements; i++) {
        cpuData[i].real = h_temp[i].x;
        cpuData[i].imag = h_temp[i].y;
    }
    free(h_temp);
    
    // Put in workspace
    mexPutVariable("base", varName, cpuArray);
    
    mxDestroyArray(cpuArray);
    mexPrintf("  Done! Variable '%s' is now in MATLAB workspace.\n", varName);
}

void write4DToWorkspace(const char* varName, const mwSize *dims, cuFloatComplex* d_data)
{
    size_t numElements = dims[0] * dims[1]  * dims[2]  * dims[3];
    
    mexPrintf("Writing %s [%zu x %zu x %zu x %zu] to workspace...\n", varName, dims[0], dims[1], dims[2], dims[3]);
    
    // Allocate and copy
    cuFloatComplex *h_temp = (cuFloatComplex*)malloc(numElements * sizeof(cuFloatComplex));
    CUDA_CHECK(cudaMemcpy(h_temp, d_data, numElements * sizeof(cuFloatComplex), 
                          cudaMemcpyDeviceToHost));
    
    // Create mxArray
    mxArray *cpuArray = mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxCOMPLEX);
    mxComplexSingle *cpuData = mxGetComplexSingles(cpuArray);
    
    // Convert
    for (size_t i = 0; i < numElements; i++) {
        cpuData[i].real = h_temp[i].x;
        cpuData[i].imag = h_temp[i].y;
    }
    free(h_temp);
    
    // Put in workspace
    mexPutVariable("base", varName, cpuArray);
    
    mxDestroyArray(cpuArray);
    mexPrintf("  Done! Variable '%s' is now in MATLAB workspace.\n", varName);
}
// ============================================================================
// DEBUG: Write matrix to MAT file
// ============================================================================
void writeMatmex(const mwSize *dims, cuFloatComplex* d_data)
{
    // Create CPU mxArray for complex single
    size_t numElements = dims[0] * dims[1];
    mxArray *cpuArray = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxCOMPLEX);
    
    // Get pointer to CPU data
    mxComplexSingle *cpuData = mxGetComplexSingles(cpuArray);
    
    // Copy from GPU to CPU
    cuFloatComplex *h_temp = (cuFloatComplex*)malloc(numElements * sizeof(cuFloatComplex));
    cudaMemcpy(h_temp, d_data, numElements * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    
    // Convert cuFloatComplex to mxComplexSingle
    for (size_t i = 0; i < numElements; i++) {
        cpuData[i].real = h_temp[i].x;
        cpuData[i].imag = h_temp[i].y;
    }
    free(h_temp);    
    
    // Open MAT-file for writing
    MATFile *pmat = matOpen("outputDebug.mat", "w");
    if (pmat == NULL) {
        mexErrMsgIdAndTxt("MATLAB:matfile", "Error creating MAT-file");
    }
    mexPrintf("Creating MAT-file Done\n");
    
    // Write the GPU array to file
    int status = matPutVariable(pmat, "myGPUArray", cpuArray);
    if (status != 0) {
        mexErrMsgIdAndTxt("MATLAB:matfile", "Error writing variable");
    }
    mexPrintf("writing MAT-file Done\n");
     
    // Close the file
    if (matClose(pmat) != 0) {
        mexErrMsgIdAndTxt("MATLAB:matfile", "Error closing file");
    }

    mexPrintf("GPU array saved to outputDebug.mat\n");
    
    // Clean up
    mxDestroyArray(cpuArray);
}

// ============================================================================
// MEX GATEWAY FUNCTION
// ============================================================================
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check number of inputs
    if (nrhs != 8) {
        mexErrMsgIdAndTxt("GRAPPA:input", 
            "Eight inputs required: k_composite_gpu, k_R_gpu, ACS_composite_gpu, "
            "ACS_gpu, mask_R, NetKTV, Ry, Rz");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("GRAPPA:output", "One output required");
    }
    
    // Initialize GPU
   // Initialize GPU
    mxInitGPU();
    
    // Check CUDA device properties
	bool bDebug=0;
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    mexPrintf("CUDA Device: %s\n", prop.name);
    mexPrintf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    mexPrintf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    mexPrintf("  Max grid size: [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    
    // Get input GPU arrays
    mxGPUArray const *k_composite = mxGPUCreateFromMxArray(prhs[0]);
    mxGPUArray const *k_R = mxGPUCreateFromMxArray(prhs[1]);
    mxGPUArray const *ACS_composite = mxGPUCreateFromMxArray(prhs[2]);
    mxGPUArray const *ACS = mxGPUCreateFromMxArray(prhs[3]);
    double const *mask = (double const *)mxGetDoubles(prhs[4]);
    int *h_NetKTV = (int *)mxGetData(prhs[5]);
    int Ry = (int)mxGetScalar(prhs[6]);
    int Rz = (int)mxGetScalar(prhs[7]);
         
    // Create output array (copy of k_R_gpu initially)
    mxGPUArray *k_recon = mxGPUCopyFromMxArray(prhs[1]);
    
    mexPrintf("\n========================================\n");
    mexPrintf("GRAPPA 5D Reconstruction Started\n");
    mexPrintf("========================================\n");
    mexPrintf("Acceleration: Ry=%d, Rz=%d (Total R=%d)\n", Ry, Rz, Ry*Rz); 
    
    // Get dimensions from k_R_gpu
    const mwSize *dims = mxGPUGetDimensions(k_R);
    int ndims = mxGPUGetNumberOfDimensions(k_R);
    
    int Nx = dims[0];
    int Ny = dims[1];
    int Nz = dims[2];
    int Nc = dims[3];
    mexPrintf("Problem size: [%d %d %d %d]\n", Nx, Ny, Nz, Nc);
    
	const mwSize *dims_ACS = mxGPUGetDimensions(ACS);
    
    int Nx_ACS = dims_ACS[0];
    int Ny_ACS = dims_ACS[1];
    int Nz_ACS = dims_ACS[2];
    int Nc_ACS = dims_ACS[3];
    mexPrintf("ACS size: [%d %d %d %d]\n", Nx_ACS, Ny_ACS, Nz_ACS, Nc_ACS);
	
	
    const mwSize *dims2 = mxGetDimensions(prhs[4]);
    mexPrintf("Mask size: [%d %d %d]\n", dims2[0], dims2[1], dims2[2]);
    
    // Get NetKTV (host array - int32)
    const mwSize *netktv_dims = mxGetDimensions(prhs[5]);
    int patch_size  = netktv_dims[0];
    int dim_patch   = netktv_dims[1];
    int num_kernels = netktv_dims[2];
	if (num_kernels==0)num_kernels=1;
    mexPrintf("Patch size: %d, Dim kernels: %d, Num kernels: %d\n", patch_size, dim_patch, num_kernels);
    
    // *** CRITICAL: Copy NetKTV to device ***
    int *d_NetKTV;
    size_t netktv_size = patch_size * dim_patch * num_kernels * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_NetKTV, netktv_size));
    CUDA_CHECK(cudaMemcpy(d_NetKTV, h_NetKTV, netktv_size, cudaMemcpyHostToDevice));
    mexPrintf("NetKTV copied to GPU (%zu bytes)\n", netktv_size);
    
    // Create weight array
   /* mwSize Wn_dims[2] = {Nc, patch_size};
    mxGPUArray *output = mxGPUCreateGPUArray(2, Wn_dims, mxSINGLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    cuFloatComplex *d_Wn = (cuFloatComplex *)mxGPUGetData(output);*/
    cuFloatComplex *d_Wn = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Wn, Nc * patch_size * sizeof(cuFloatComplex)));
    // Get pointers to GPU data
    cuFloatComplex const *d_k_composite   = (cuFloatComplex const*)mxGPUGetDataReadOnly(k_composite);
    cuFloatComplex const *d_k_R           = (cuFloatComplex const*)mxGPUGetDataReadOnly(k_R);
    cuFloatComplex const *d_ACS_composite = (cuFloatComplex const*)mxGPUGetDataReadOnly(ACS_composite);
    cuFloatComplex const *d_ACS           = (cuFloatComplex const*)mxGPUGetDataReadOnly(ACS);
    cuFloatComplex *d_k_recon             = (cuFloatComplex*)mxGPUGetData(k_recon);
    
    // Initialize cuBLAS and cuSOLVER
    cublasHandle_t cublas_h;
    cusolverDnHandle_t cusolver_h;
    CUBLAS_CHECK(cublasCreate(&cublas_h));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_h));
     
    // Process each kernel (cK loops from 1 to Ry*Rz-1)
    int total_kernels = Ry * Rz-1;
    
    for (int cK = 0; cK < total_kernels; cK++) 
    { 
        mexPrintf("\n========================================\n");
        mexPrintf("Processing kernel %d/%d\n", cK, total_kernels - 1);
        mexPrintf("========================================\n");
        
        // === TRAINING PHASE ===
        mexPrintf("\n--- TRAINING PHASE ---\n");
        
        int *d_train_pts;
        int num_train = 0;
        num_train = getListOfExamples(Nx_ACS, Ny_ACS, Nz_ACS, h_NetKTV, patch_size, dim_patch, cK, &d_train_pts); 
        
        if (num_train == 0) {
            mexPrintf("No training examples found, skipping kernel %d\n", cK);
            continue;
        }

        mexPrintf("Training examples: %d\n", num_train);
        
        // Extract patches from ACS_composite (source patches)
        cuFloatComplex *d_Sn;
        CUDA_CHECK(cudaMalloc(&d_Sn, patch_size * num_train * sizeof(cuFloatComplex)));
        CUDA_CHECK(cudaMemset(d_Sn, 0, patch_size * num_train * sizeof(cuFloatComplex)));
        
        extractPatchesKernel(d_ACS_composite, d_train_pts, d_NetKTV, d_Sn, num_train, dim_patch, patch_size, cK, Nx_ACS, Ny_ACS, Nz_ACS, Nc_ACS);
        mexPrintf("extractPatchesKernel Done\n");
        
        // Debug: Write Sn to file
        if (cK == 0 && bDebug) {
            mwSize Sn_dims[2] = {patch_size, num_train};
			writeToWorkspace("debug_Sn", Sn_dims, d_Sn);
        }
        
        // Extract target values from ACS (target patches)
        cuFloatComplex *d_Xn;
        CUDA_CHECK(cudaMalloc(&d_Xn, Nc * num_train * sizeof(cuFloatComplex)));
        CUDA_CHECK(cudaMemset(d_Xn, 0, Nc * num_train * sizeof(cuFloatComplex)));
        
        extractACSKernel(d_ACS, d_train_pts, d_Xn, num_train, Nx_ACS, Ny_ACS, Nz_ACS, Nc_ACS);
        mexPrintf("extractACSKernel Done\n");
        
		// Debug: Write Xn to file
        if (cK == 0 && bDebug) {
            mwSize Xn_dims[2] = {Nc, num_train};
			writeToWorkspace("debug_Xn", Xn_dims, d_Xn);
        }
		
        // Solve for GRAPPA weights
        //  cudaFree(d_Wn);
        //  CUDA_CHECK(cudaMalloc(&d_Wn, Nc * patch_size * sizeof(cuFloatComplex)));
        solveGRAPPAWeights(cublas_h, cusolver_h, d_Xn, d_Sn, d_Wn, Nc, patch_size, num_train);
        mexPrintf("solveGRAPPAWeights Done\n");

		// Debug: Write Wn to file
        if (cK == 0 && bDebug) {
            mwSize Wn_dims[2] = {Nc, patch_size};
			writeToWorkspace("debug_Wn", Wn_dims, d_Wn);
        }
		
        // === RECONSTRUCTION PHASE ===
        mexPrintf("\n--- RECONSTRUCTION PHASE ---\n");
        
        int recon_count = 0;
        int *d_recon_pts_temp;
        //CUDA_CHECK(cudaMalloc(&d_recon_pts_temp, Nx * Ny * Nz * 3 * sizeof(int)));
        
        recon_count = findReconPointsKernel(mask, &d_recon_pts_temp, cK+1, Nx, Ny, Nz);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        if (recon_count > 0) 
        {
            mexPrintf("Reconstruction points: %d\n", recon_count);
            
            // Copy valid reconstruction points
            int *d_recon_pts;
            CUDA_CHECK(cudaMalloc(&d_recon_pts, recon_count * 3 * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_recon_pts, d_recon_pts_temp, recon_count * 3 * sizeof(int), cudaMemcpyDeviceToDevice));
            
            // Check if we need to segment the reconstruction
            size_t bytes_needed = (size_t)patch_size * recon_count * sizeof(cuFloatComplex);
            float gb_needed = bytes_needed / (1024.0f * 1024.0f * 1024.0f);
            
            
			// Process all at once
			mexPrintf("Processing all points at once (%.2f GB)\n", gb_needed);
			
			cuFloatComplex *d_Sr;
			CUDA_CHECK(cudaMalloc(&d_Sr, patch_size * recon_count * sizeof(cuFloatComplex)));
			CUDA_CHECK(cudaMemset(d_Sr, 0, patch_size * recon_count * sizeof(cuFloatComplex)));
			
			extractPatchesKernel(d_k_composite, d_recon_pts, d_NetKTV, d_Sr, recon_count, dim_patch, patch_size, cK, Nx, Ny, Nz, Nc);
				// Debug: Write Sr to file
			if (cK == 0 && bDebug) {
				mwSize Dr_dims[2] = {patch_size, recon_count};
				writeToWorkspace("debug_Sr", Dr_dims, d_Sr);
			}
			
			// Apply weights
			cuFloatComplex *d_Xr;
			CUDA_CHECK(cudaMalloc(&d_Xr, Nc * recon_count * sizeof(cuFloatComplex)));
			CUDA_CHECK(cudaMemset(d_Xr, 0, Nc * recon_count * sizeof(cuFloatComplex)));
			
			cuFloatComplex alpha = make_cuFloatComplex(1.0f, 0.0f);
			cuFloatComplex beta = make_cuFloatComplex(0.0f, 0.0f);
			
			CUBLAS_CHECK(cublasCgemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, 
									Nc, recon_count, patch_size,
									&alpha, d_Wn, Nc, d_Sr, patch_size,
									&beta, d_Xr, Nc));
			
			// Assign to output
			assignReconValuesKernel(d_k_recon, d_Xr, d_recon_pts, recon_count,Nx, Ny, Nz, Nc);
			
			cudaFree(d_Sr);
			cudaFree(d_Xr);
            cudaFree(d_recon_pts);
        } 
        else 
        {
            mexPrintf("No reconstruction points found for kernel %d\n", cK);
        }
        
        // Cleanup this kernel's data
        cudaFree(d_train_pts);
        cudaFree(d_Sn);
        cudaFree(d_Xn);
        cudaFree(d_recon_pts_temp);
    }
    
    mexPrintf("\n========================================\n");
    mexPrintf("Reconstruction Complete!\n");
    mexPrintf("========================================\n");
    
    // Cleanup
    cudaFree(d_NetKTV);  // Don't forget to free this!
    cublasDestroy(cublas_h);
    cusolverDnDestroy(cusolver_h);
    
    // Return output
    plhs[0] = mxGPUCreateMxArrayOnGPU(k_recon);
   //mxGPUDestroyGPUArray(output);
    
    cudaFree(d_Wn);
    
    // Cleanup input arrays
    mxGPUDestroyGPUArray(k_composite);
    mxGPUDestroyGPUArray(k_R);
    mxGPUDestroyGPUArray(ACS_composite);
    mxGPUDestroyGPUArray(ACS);
    mxGPUDestroyGPUArray(k_recon);
}