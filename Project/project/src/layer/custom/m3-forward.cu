#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define MAX_BATCH_SIZE 1000
#define MAX_MASK_SIZE 8192  // Adjust based on the maximum expected mask size
#define KERNEL_SIZE 7       // Assuming maximum K=7 for unrolling; adjust as needed

// Declare constant memory for the mask
__constant__ float const_mask[MAX_MASK_SIZE];

// Optimized convolution kernel
__global__ void conv_forward_kernel(const float *__restrict__ input, float *__restrict__ output,
                                    const int Batch, const int Map_out, const int Channel,
                                    const int Height, const int Width, const int K) {
    // Define shared memory for input tile
    extern __shared__ float shared_input[];

    // Calculate output dimensions
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Calculate thread indices
    int tx = threadIdx.x;  // Width index within the tile
    int ty = threadIdx.y;  // Height index within the tile

    // Calculate output indices
    int b = blockIdx.z;         // Batch index
    int m = blockIdx.y;         // Output feature map index (Map_out dimension)
    int h_out = blockIdx.x * TILE_WIDTH + ty;  // Output height index
    int w_out = tx;                           // Output width index

    // Shared memory dimensions
    int shared_size = TILE_WIDTH + K - 1;

    // Only proceed if within output bounds
    if (h_out < Height_out && w_out < Width_out) {
        float acc = 0.0f;

        // Load input tile into shared memory
        for (int c = 0; c < Channel; ++c) {
            // Calculate input indices
            int h_in = h_out;
            int w_in = w_out;

            // Load the shared memory tile
            for (int i = ty; i < shared_size; i += TILE_WIDTH) {
                for (int j = tx; j < shared_size; j += TILE_WIDTH) {
                    int in_h = h_in + i - ty;
                    int in_w = w_in + j - tx;
                    if (in_h < Height && in_w < Width) {
                        shared_input[i * shared_size + j] = input[b * (Channel * Height * Width) +
                                                                  c * (Height * Width) +
                                                                  in_h * Width + in_w];
                    } else {
                        shared_input[i * shared_size + j] = 0.0f;
                    }
                }
            }
            __syncthreads();

            // Perform convolution using the shared input and constant mask
            #pragma unroll
            for (int p = 0; p < K; ++p) {
                #pragma unroll
                for (int q = 0; q < K; ++q) {
                    float mask_value = const_mask[m * (Channel * K * K) + c * (K * K) + p * K + q];
                    float input_value = shared_input[(ty + p) * shared_size + (tx + q)];
                    acc += input_value * mask_value;
                }
            }
            __syncthreads();  // Ensure all threads have completed using shared memory
        }

        // Write the result to the output tensor
        output[b * (Map_out * Height_out * Width_out) +
               m * (Height_out * Width_out) +
               h_out * Width_out +
               w_out] = acc;
    }
}

// Host function to allocate memory and copy data to the GPU (Prolog)
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input,
                                                    const float *host_mask, float **device_output_ptr,
                                                    float **device_input_ptr, float **device_mask_ptr,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K) {
    // Allocate device memory for input
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    cudaMalloc((void **)device_input_ptr, input_size);
    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);

    // Allocate device memory for output
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);
    cudaMalloc((void **)device_output_ptr, output_size);

    // Copy the mask to constant memory
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);
    cudaMemcpyToSymbol(const_mask, host_mask, mask_size);

    // Note: device_mask_ptr is not used since we are using constant memory for the mask

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA error (prolog): " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

// Host function to launch the kernel
__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input,
                                             const float *device_mask, const int Batch, const int Map_out,
                                             const int Channel, const int Height, const int Width, const int K) {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Configure block and grid dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((Height_out + TILE_WIDTH - 1) / TILE_WIDTH,  // Number of vertical tiles
                 Map_out,                                     // One block per output feature map
                 Batch);                                      // One block per batch

    // Calculate shared memory size
    size_t shared_mem_size = (TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) * sizeof(float);

    // Launch the kernel
    conv_forward_kernel<<<dimGrid, dimBlock, shared_mem_size>>>(
        device_input, device_output, Batch, Map_out, Channel, Height, Width, K);

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA error (conv_forward_gpu): " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }

    // Synchronize to ensure all kernels have finished
    cudaDeviceSynchronize();
}

// Host function to copy the output back to the host and free device memory (Epilog)
__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output,
                                                    float *device_input, float *device_mask,
                                                    const int Batch, const int Map_out, const int Channel,
                                                    const int Height, const int Width, const int K) {
    // Copy the output back to the host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    // Note: device_mask is not used since we are using constant memory

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA error (epilog): " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

// Function to get device properties (unchanged)
__host__ void GPUInterface::get_device_properties() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1]
                  << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1]
                  << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}
