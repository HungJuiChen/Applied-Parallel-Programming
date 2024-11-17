#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_runtime.h>

#include <mma.h> // For Tensor Cores (req_1)

using namespace nvcuda; // For WMMA (req_1)

#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define MAX_BATCH_SIZE 1000

// ========================================================================
// Requirement 2: Kernel Fusion
// Fused kernel that combines matrix unrolling, matrix multiplication, and permutation.

__global__ void conv_forward_kernel_fused(
    const float *input, const float *mask, float *output,
    const int Batch, const int Map_out, const int Channel,
    const int Height, const int Width, const int K)
{
    // Compute output dimensions
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Calculate global thread indices
    int b = blockIdx.z * blockDim.z + threadIdx.z;       // Batch index
    int m = blockIdx.y * blockDim.y + threadIdx.y;       // Output feature map
    int h = blockIdx.x * blockDim.x + threadIdx.x;       // Output height
    int w = threadIdx.w;                                 // Output width (using threadIdx.w since blockDim.w is set)

    if (b < Batch && m < Map_out && h < Height_out && w < Width_out) {
        float value = 0.0f;

        // Perform convolution
        for (int c = 0; c < Channel; ++c) {
            for (int p = 0; p < K; ++p) {
                for (int q = 0; q < K; ++q) {
                    int h_in = h + p;
                    int w_in = w + q;
                    float input_value = input[b * Channel * Height * Width +
                                              c * Height * Width +
                                              h_in * Width + w_in];
                    float mask_value = mask[m * Channel * K * K +
                                            c * K * K +
                                            p * K + q];
                    value += input_value * mask_value;
                }
            }
        }

        // Store the result
        output[b * Map_out * Height_out * Width_out +
               m * Height_out * Width_out +
               h * Width_out + w] = value;
    }
}

// ========================================================================
// Requirement 1: Tensor Cores
// Modified matrix multiplication kernel to use WMMA API

__global__ void matrixMultiplyWMMA(const float *A, const float *B, float *C,
                                   int M, int N, int K)
{
    // Leading dimensions
    int lda = K;
    int ldb = N;
    int ldc = N;

    // Tile indices
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);

    if (warpM * 16 >= M || warpN * 16 >= N) return;

    // Fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, float, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, float, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    int aRow = warpM * 16;
    int bCol = warpN * 16;

    for (int k = 0; k < K; k += 16) {
        if (aRow < M && k < K) {
            wmma::load_matrix_sync(a_frag, A + aRow * lda + k, lda);
        } else {
            wmma::fill_fragment(a_frag, 0.0f);
        }

        if (bCol < N && k < K) {
            wmma::load_matrix_sync(b_frag, B + k * ldb + bCol, ldb);
        } else {
            wmma::fill_fragment(b_frag, 0.0f);
        }

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store the output
    if (aRow < M && bCol < N) {
        wmma::store_matrix_sync(C + aRow * ldc + bCol, c_frag, ldc, wmma::mem_row_major);
    }
}

__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    //(void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    //(void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    const int H_unroll = Channel * K * K;
    const int W_unroll = Batch * Height_out * Width_out;

    // Calculate h_unroll and w_unroll using 2D grid and block indices
    int h_unroll = blockIdx.y * blockDim.y + threadIdx.y;
    int w_unroll = blockIdx.x * blockDim.x + threadIdx.x;

    if (h_unroll < H_unroll && w_unroll < W_unroll) {
        int c = h_unroll / (K * K);
        int p = (h_unroll % (K * K)) / K;
        int q = (h_unroll % (K * K)) % K;

        int b = w_unroll / (Height_out * Width_out);
        int remainder = w_unroll % (Height_out * Width_out);
        int h = remainder / Width_out;
        int w = remainder % Width_out;

        int input_row = h + p;
        int input_col = w + q;
        
    
        // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
        // An example use of these macros:
        // float a = in_4d(0,0,0,0)

        #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
        #define out_2d(i1, i0) output[(i1) * W_unroll + (i0)]

        // TODO: Insert your input matrix unrolling kernel code here
        if (input_row < Height && input_col < Width) {
            out_2d(h_unroll, w_unroll) = in_4d(b, c, input_row, input_col);
        } else {
            out_2d(h_unroll, w_unroll) = 0.0f;
        }

        #undef in_4d
        #undef out_2d
    }
}

// Tiled matrix multiplication kernel. Computes C = AB
// You don't need to modify this kernel.
__global__ void matrixMultiplyShared(const float *A, const float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    float val = 0;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = A[(size_t) row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0;
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[((size_t) tileId * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            tileB[ty][tx] = 0;
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = val;
    }
}

// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}

// Requirement 0: Streams
// The main computation is moved into conv_forward_gpu_prolog to manage streams and overlap data transfer with computation.

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }

    // Allocate device memory for the mask
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);
    cudaMalloc((void**) device_mask_ptr, mask_size);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    const int num_batches = (Batch + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;
    const int num_streams = 4; // Adjust based on GPU capabilities

    // Create CUDA streams
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Allocate device buffers per stream
    size_t input_size_per_batch = MAX_BATCH_SIZE * Channel * Height * Width * sizeof(float);
    size_t output_size_per_batch = MAX_BATCH_SIZE * Map_out * Height_out * Width_out * sizeof(float);
    float* device_input_buffers[num_streams];
    float* device_output_buffers[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaMalloc((void**)&device_input_buffers[i], input_size_per_batch);
        cudaMalloc((void**)&device_output_buffers[i], output_size_per_batch);
    }

    // Loop over batches
    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        int current_batch_size = (batch_idx == num_batches - 1) ?
                                 (Batch - batch_idx * MAX_BATCH_SIZE) : MAX_BATCH_SIZE;
        int stream_id = batch_idx % num_streams;

        const float* host_input_batch = host_input + batch_idx * MAX_BATCH_SIZE * Channel * Height * Width;
        float* host_output_batch = const_cast<float*>(host_output) +
                                   batch_idx * MAX_BATCH_SIZE * Map_out * Height_out * Width_out;

        // Asynchronous copy of input data to device
        size_t current_input_size = current_batch_size * Channel * Height * Width * sizeof(float);
        cudaMemcpyAsync(device_input_buffers[stream_id], host_input_batch, current_input_size,
                        cudaMemcpyHostToDevice, streams[stream_id]);

        // Launch convolution kernel
        conv_forward_gpu(device_output_buffers[stream_id], device_input_buffers[stream_id],
                         *device_mask_ptr, current_batch_size, Map_out, Channel,
                         Height, Width, K, streams[stream_id]);

        // Asynchronous copy of output data back to host
        size_t current_output_size = current_batch_size * Map_out * Height_out * Width_out * sizeof(float);
        cudaMemcpyAsync(host_output_batch, device_output_buffers[stream_id],
                        current_output_size, cudaMemcpyDeviceToHost, streams[stream_id]);
    }

    // Synchronize and destroy streams
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Free device memory
    for (int i = 0; i < num_streams; ++i) {
        cudaFree(device_input_buffers[i]);
        cudaFree(device_output_buffers[i]);
    }

    cudaFree(*device_mask_ptr);

    // Nullify device pointers
    *device_input_ptr = nullptr;
    *device_output_ptr = nullptr;
    *device_mask_ptr = nullptr;

    // Error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error (prolog): "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    // const int Height_unrolled = Channel * K * K;
    // const int W_unrolled = Batch * Height_out * Width_out;

    // // Determine the number of mini-batches
    // int num_batches = (Batch + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;

    // float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    // float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    
    // //cudaMalloc((void**)&unrolled_matrix, (size_t) Batch * Channel * K * K * Height_out * Width_out * sizeof(float));
    // //cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));
    
    // // Allocate device memory for unrolled_matrix and matmul_output for the maximum mini-batch size
    // size_t max_unroll_size = Height_unrolled * (MAX_BATCH_SIZE * Height_out * Width_out) * sizeof(float);
    // cudaMalloc((void**)&unrolled_matrix, max_unroll_size);

    // size_t max_matmul_size = Map_out * (MAX_BATCH_SIZE * Height_out * Width_out) * sizeof(float);
    
    // cudaMalloc((void**)&matmul_output, max_matmul_size);
    // // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    
    // // Iterate over each mini-batch
    // for(int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {

    //     // Determine the current stream
    //     int stream_id = batch_idx % NUM_STREAMS;
    //     cudaStream_t stream = streams[stream_id];

    //     // Calculate the current mini-batch size
    //     int current_batch_size = (batch_idx == num_batches - 1) ? (Batch - batch_idx * MAX_BATCH_SIZE) : MAX_BATCH_SIZE;

    //     // Calculate current W_unroll
    //     int current_W_unroll = current_batch_size * Height_out * Width_out;
    //     // Set the kernel dimensions for unrolling using a 2D grid
    //     dim3 blockDim_unroll(16, 16);
    //     dim3 gridDim_unroll((current_W_unroll + blockDim_unroll.x - 1) / blockDim_unroll.x,
    //                         (Height_unrolled + blockDim_unroll.y - 1) / blockDim_unroll.y);
        
    //     // Call the matrix unrolling kernel for the current mini-batch
    //     // matrix_unrolling_kernel<<<gridDim_unroll, blockDim_unroll>>>(
    //     //     device_input + batch_idx * MAX_BATCH_SIZE * Channel * Height * Width, // Offset input pointer
    //     //     unrolled_matrix, 
    //     //     current_batch_size, 
    //     //     Channel, 
    //     //     Height, 
    //     //     Width, 
    //     //     K
    //     // );

    //     // Asynchronously call the matrix unrolling kernel on the current stream
    //     matrix_unrolling_kernel<<<gridDim_unroll, blockDim_unroll, 0, stream>>>(
    //         device_input + batch_idx * MAX_BATCH_SIZE * Channel * Height * Width, // Offset input pointer
    //         unrolled_matrix, 
    //         current_batch_size, 
    //         Channel, 
    //         Height, 
    //         Width, 
    //         K
    //     );

        
    //     cudaError_t error = cudaGetLastError();
    //     if(error != cudaSuccess)
    //     {
    //         std::cout<<"CUDA error (unrolling kernel): "<<cudaGetErrorString(error)<<std::endl;
    //         exit(-1);
    //     }

    //     // TODO: Set the kernel dimensions and call the matmul kernel
    //     int numARows = Map_out;
    //     int numAColumns = Channel * K * K;
    //     int numBRows = Channel * K * K;
    //     int numBColumns = current_W_unroll;
    //     int numCRows = Map_out;
    //     int numCColumns = current_W_unroll;

    //     dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    //     dim3 dimGrid((numCColumns - 1)/TILE_WIDTH + 1, (numCRows -1)/TILE_WIDTH + 1);

    //     // // Call the matrix multiplication kernel
    //     // matrixMultiplyShared<<<dimGrid, dimBlock>>>(device_mask, unrolled_matrix, matmul_output,
    //     //                                             numARows, numAColumns,
    //     //                                             numBRows, numBColumns,
    //     //                                             numCRows, numCColumns);

    //     // Asynchronously call the matrix multiplication kernel on the current stream
    //     matrixMultiplyShared<<<dimGrid, dimBlock, 0, stream>>>(
    //         device_mask, unrolled_matrix, matmul_output,
    //         numARows, numAColumns,
    //         numBRows, numBColumns,
    //         numCRows, numCColumns
    //     );
        
    //     error = cudaGetLastError();
    //     if(error != cudaSuccess)
    //     {
    //         std::cout<<"CUDA error (matmul kernel): "<<cudaGetErrorString(error)<<std::endl;
    //         exit(-1);
    //     }

    //     // Permute the result of matrix multiplication
    //     const int out_image_size = Height_out * Width_out;
    //     dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, current_batch_size, 1);
    //     // matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
    //     //     matmul_output, device_output, Map_out, Batch, out_image_size
    //     // );
    //     // matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
    //     //     matmul_output, 
    //     //     device_output + batch_idx * MAX_BATCH_SIZE * Map_out * out_image_size, // Offset output pointer
    //     //     Map_out, 
    //     //     current_batch_size, 
    //     //     out_image_size
    //     // );

    //     // Asynchronously call the permutation kernel on the current stream
    //     matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE, 0, stream>>>(
    //         matmul_output, 
    //         device_output + batch_idx * MAX_BATCH_SIZE * Map_out * out_image_size, // Offset output pointer
    //         Map_out, 
    //         current_batch_size, 
    //         out_image_size
    //     );

    //     // Check for errors after permutation
    //     error = cudaGetLastError();
    //     if(error != cudaSuccess)
    //     {
    //         std::cout<<"CUDA error (permute kernel): "<<cudaGetErrorString(error)<<std::endl;
    //         exit(-1);
    //     }
    // }

    // // Synchronize all streams to ensure all operations are complete
    // for(int i = 0; i < NUM_STREAMS; ++i){
    //     cudaStreamSynchronize(streams[i]);
    // }

    // cudaFree(matmul_output);
    // cudaFree(unrolled_matrix);

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Decide whether to use the fused kernel (req_2) or the WMMA kernel (req_1)
    // For the purpose of this code, we'll demonstrate using the WMMA kernel (req_1)

    const int Height_unrolled = Channel * K * K;
    int W_unroll = Batch * Height_out * Width_out;

    // Allocate device memory for unrolled_matrix and matmul_output
    size_t unroll_size = Height_unrolled * W_unroll * sizeof(float);
    float *unrolled_matrix;
    cudaMalloc((void**)&unrolled_matrix, unroll_size);

    size_t matmul_size = Map_out * W_unroll * sizeof(float);
    float *matmul_output;
    cudaMalloc((void**)&matmul_output, matmul_size);

    // Set the kernel dimensions for unrolling
    dim3 blockDim_unroll(16, 16);
    dim3 gridDim_unroll((W_unroll + blockDim_unroll.x - 1) / blockDim_unroll.x,
                        (Height_unrolled + blockDim_unroll.y - 1) / blockDim_unroll.y);

    // Call the matrix unrolling kernel (assuming it's defined elsewhere)
    matrix_unrolling_kernel<<<gridDim_unroll, blockDim_unroll, 0, stream>>>(
        device_input, unrolled_matrix, Batch, Channel, Height, Width, K);

    // Error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error (unrolling kernel): "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    // Prepare dimensions for WMMA
    int M = Map_out;
    int N = W_unroll;
    int K_wmma = Height_unrolled;

    // Pad M, N, K to multiples of 16
    int M_aligned = ((M + 15)/16)*16;
    int N_aligned = ((N + 15)/16)*16;
    int K_aligned = ((K_wmma + 15)/16)*16;

    // Set the kernel dimensions for WMMA matmul
    dim3 dimBlock(16, 16);
    dim3 dimGrid((N_aligned + 16 * 16 - 1) / (16 * 16), (M_aligned + 16 - 1) / 16);

    // Call the WMMA matrix multiplication kernel
    matrixMultiplyWMMA<<<dimGrid, dimBlock, 0, stream>>>(
        device_mask, unrolled_matrix, matmul_output,
        M_aligned, N_aligned, K_aligned);

    // Error checking
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error (WMMA matmul kernel): "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    // Permute the result
    const int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE, 0, stream>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size);

    // Error checking
    error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error (permute kernel): "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

    // Free device memory
    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);
    //cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);
    
    // Asynchronously copy output from device to host using stream 0
    cudaMemcpyAsync(host_output, device_output, output_size, cudaMemcpyDeviceToHost, streams[0]);

    // Synchronize all streams to ensure copy is complete
    for(int i = 0; i < NUM_STREAMS; ++i){
        cudaStreamSynchronize(streams[i]);
    }

    // TODO: Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

    // Free pinned host memory
    cudaFreeHost(host_input_pinned);
    cudaFreeHost(host_mask_pinned);
    cudaFreeHost(host_output_pinned);

    // Destroy CUDA Streams
    for(int i = 0; i < NUM_STREAMS; ++i){
        cudaStreamDestroy(streams[i]);
    }

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error (epilog): "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}