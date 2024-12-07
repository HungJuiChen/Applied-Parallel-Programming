#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define MAX_BATCH_SIZE 1000

__global__ void fused_conv_kernel(const float *__restrict__ input, const float *__restrict__ mask, float *__restrict__ output,
                                  const int Batch, const int Map_out, const int Channel,
                                  const int Height, const int Width, const int K) {
    const int TILE_WIDTH = 16; // Tile width
    const int REG_TILE = 4;    // Register tile size per thread
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Shared memory allocation for a tile
    __shared__ float shared_input[TILE_WIDTH + K - 1][TILE_WIDTH + K - 1];
    __shared__ float shared_mask[K][K];

    // Register tiles for storing partial results
    float reg_tile[REG_TILE][REG_TILE] = {0};

    // Thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    // Load shared memory tiles
    for (int c = 0; c < Channel; ++c) {
        // Load input tile into shared memory
        for (int i = 0; i < K; ++i) {
            for (int j = 0; j < K; ++j) {
                int input_row = row + i;
                int input_col = col + j;
                if (input_row < Height && input_col < Width) {
                    shared_input[ty + i][tx + j] = input[c * Height * Width + input_row * Width + input_col];
                } else {
                    shared_input[ty + i][tx + j] = 0.0f;
                }
            }
        }

        // Load mask into shared memory
        for (int i = ty; i < K; i += TILE_WIDTH) {
            for (int j = tx; j < K; j += TILE_WIDTH) {
                shared_mask[i][j] = mask[c * K * K + i * K + j];
            }
        }

        __syncthreads();

        // Perform convolution with the tile
        for (int i = 0; i < REG_TILE; ++i) {
            for (int j = 0; j < REG_TILE; ++j) {
                int r = ty * REG_TILE + i;
                int s = tx * REG_TILE + j;
                for (int p = 0; p < K; ++p) {
                    for (int q = 0; q < K; ++q) {
                        reg_tile[i][j] += shared_input[r + p][s + q] * shared_mask[p][q];
                    }
                }
            }
        }

        __syncthreads();
    }

    // Write results back to global memory
    for (int i = 0; i < REG_TILE; ++i) {
        for (int j = 0; j < REG_TILE; ++j) {
            int output_row = row + i;
            int output_col = col + j;
            if (output_row < Height_out && output_col < Width_out) {
                output[blockIdx.z * Map_out * Height_out * Width_out +
                       row * Width_out + col] = reg_tile[i][j];
            }
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // Allocate device memory for input
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    cudaMalloc((void**) device_input_ptr, input_size);
    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);

    // Allocate device memory for output
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);
    cudaMalloc((void**) device_output_ptr, output_size);

    // Allocate device memory for mask
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);
    cudaMalloc((void**) device_mask_ptr, mask_size);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);

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
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Determine the number of mini-batches
    int num_batches = (Batch + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;

    // Iterate over each mini-batch
    for(int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        // Calculate the current mini-batch size
        int current_batch_size = (batch_idx == num_batches - 1) ? (Batch - batch_idx * MAX_BATCH_SIZE) : MAX_BATCH_SIZE;
        int current_W_unroll = current_batch_size * Height_out * Width_out;

        // Set grid and block dimensions
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((current_W_unroll + TILE_WIDTH - 1) / TILE_WIDTH,
                     (Map_out + TILE_WIDTH - 1) / TILE_WIDTH);

        // Launch the fused kernel
        fused_conv_kernel<<<dimGrid, dimBlock>>>(
            device_input + batch_idx * MAX_BATCH_SIZE * Channel * Height * Width,
            device_mask,
            device_output + batch_idx * MAX_BATCH_SIZE * Map_out * Height_out * Width_out,
            current_batch_size,
            Map_out,
            Channel,
            Height,
            Width,
            K
        );

        // Error checking
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess) {
            std::cout << "CUDA error (fused kernel): " << cudaGetErrorString(error) << std::endl;
            exit(-1);
        }
    }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);
    
    // TODO: Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

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