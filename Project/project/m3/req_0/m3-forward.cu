#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define MAX_BATCH_SIZE 1000

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

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate device memory for the mask and copy it
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);
    cudaMalloc((void**) device_mask_ptr, mask_size);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);

    // Determine the batch slice size and number of slices
    int batch_slice_size = MAX_BATCH_SIZE; // Adjust based on available memory
    int num_slices = (Batch + batch_slice_size - 1) / batch_slice_size;

    // Compute output dimensions
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Allocate device memory for input and output slices
    size_t input_slice_size = batch_slice_size * Channel * Height * Width * sizeof(float);
    cudaMalloc((void**) device_input_ptr, input_slice_size); // device_input_slices

    size_t output_slice_size = batch_slice_size * Map_out * Height_out * Width_out * sizeof(float);
    cudaMalloc((void**) device_output_ptr, output_slice_size); // device_output_slices

    // Allocate device memory for unrolled matrix and matmul output
    int H_unroll = Channel * K * K;
    int W_unroll = batch_slice_size * Height_out * Width_out;
    size_t unroll_size = H_unroll * W_unroll * sizeof(float);
    float *unrolled_matrix;
    cudaMalloc((void**)&unrolled_matrix, unroll_size);

    int numCRows = Map_out;
    int numCColumns = batch_slice_size * Height_out * Width_out;
    size_t matmul_size = numCRows * numCColumns * sizeof(float);
    float *matmul_output;
    cudaMalloc((void**)&matmul_output, matmul_size);

    // Create CUDA streams
    cudaStream_t *streams = new cudaStream_t[num_slices];
    for (int i = 0; i < num_slices; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Process each batch slice
    for (int slice_idx = 0; slice_idx < num_slices; ++slice_idx) {
        int current_batch_size = (slice_idx == num_slices - 1) ?
            (Batch - slice_idx * batch_slice_size) : batch_slice_size;

        // Compute host pointers for the current slice
        const float *host_input_slice = host_input + slice_idx * batch_slice_size * Channel * Height * Width;
        float *host_output_slice = const_cast<float*>(host_output) + slice_idx * batch_slice_size * Map_out * Height_out * Width_out;

        // Asynchronously copy input slice to device
        size_t current_input_size = current_batch_size * Channel * Height * Width * sizeof(float);
        cudaMemcpyAsync(*device_input_ptr, host_input_slice, current_input_size, cudaMemcpyHostToDevice, streams[slice_idx]);

        // Update dimensions for the current batch size
        int current_W_unroll = current_batch_size * Height_out * Width_out;

        // Set kernel dimensions for unrolling
        dim3 blockDim_unroll(16, 16);
        dim3 gridDim_unroll((current_W_unroll + blockDim_unroll.x - 1) / blockDim_unroll.x,
                            (H_unroll + blockDim_unroll.y - 1) / blockDim_unroll.y);

        // Launch unrolling kernel
        matrix_unrolling_kernel<<<gridDim_unroll, blockDim_unroll, 0, streams[slice_idx]>>>(
            *device_input_ptr,
            unrolled_matrix,
            current_batch_size,
            Channel,
            Height,
            Width,
            K
        );

        // Set kernel dimensions for matrix multiplication
        int numARows = Map_out;
        int numAColumns = Channel * K * K;
        int numBRows = Channel * K * K;
        int numBColumns = current_W_unroll;
        numCRows = Map_out;
        numCColumns = current_W_unroll;

        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((numCColumns - 1)/TILE_WIDTH + 1, (numCRows -1)/TILE_WIDTH + 1);

        // Launch matrix multiplication kernel
        matrixMultiplyShared<<<dimGrid, dimBlock, 0, streams[slice_idx]>>>(
            *device_mask_ptr,
            unrolled_matrix,
            matmul_output,
            numARows, numAColumns,
            numBRows, numBColumns,
            numCRows, numCColumns
        );

        // Set kernel dimensions for permutation
        const int out_image_size = Height_out * Width_out;
        dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, current_batch_size, 1);

        // Launch permutation kernel
        matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE, 0, streams[slice_idx]>>>(
            matmul_output,
            *device_output_ptr,
            Map_out,
            current_batch_size,
            out_image_size
        );

        // Asynchronously copy output slice back to host
        size_t current_output_size = current_batch_size * Map_out * Height_out * Width_out * sizeof(float);
        cudaMemcpyAsync(host_output_slice, *device_output_ptr, current_output_size, cudaMemcpyDeviceToHost, streams[slice_idx]);
    }

    // Synchronize and destroy streams
    for (int i = 0; i < num_slices; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;

    // Free device memory
    cudaFree(unrolled_matrix);
    cudaFree(matmul_output);
    cudaFree(*device_input_ptr);
    cudaFree(*device_output_ptr);
    cudaFree(*device_mask_ptr);

    // Set device pointers to nullptr as we've freed the memory
    *device_input_ptr = nullptr;
    *device_output_ptr = nullptr;
    *device_mask_ptr = nullptr;

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error (prolog): "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // nothing to do on this application
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
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