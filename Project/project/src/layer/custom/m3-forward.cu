#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16        // Tile width for shared memory tiling
#define MAX_BATCH_SIZE 1000  // Maximum batch size to handle
#define MAX_MASK_SIZE 8192   // Maximum mask size for constant memory (adjust as needed)

// Constant memory for the mask
__constant__ float const_mask[MAX_MASK_SIZE];

/**
 * @brief Optimized convolution kernel using shared memory and constant memory.
 *
 * @param input Pointer to the input data in device memory.
 * @param output Pointer to the output data in device memory.
 * @param Batch Number of images in the batch.
 * @param Map_out Number of output feature maps.
 * @param Channel Number of input feature maps.
 * @param Height Height of the input image.
 * @param Width Width of the input image.
 * @param K Kernel size (assumed square).
 */
__global__ void conv_forward_kernel(const float *__restrict__ input, float *__restrict__ output,
                                    const int Batch, const int Map_out, const int Channel,
                                    const int Height, const int Width, const int K) {
    // Output dimensions
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Shared memory for input tile
    extern __shared__ float shared_input[];

    // Calculate global thread indices
    int n = blockIdx.z;  // Batch index
    int m = blockIdx.y;  // Output feature map index (Map_out)
    int h_out = blockIdx.x * TILE_WIDTH + threadIdx.y;  // Output height index
    int w_out = threadIdx.x;  // Output width index

    // Check if within output bounds
    if (h_out < Height_out && w_out < Width_out) {
        float acc = 0.0f;  // Accumulator for the convolution sum

        // Loop over all input channels
        for (int c = 0; c < Channel; ++c) {
            // Load input tile into shared memory
            for (int i = threadIdx.y; i < TILE_WIDTH + K - 1; i += TILE_WIDTH) {
                for (int j = threadIdx.x; j < TILE_WIDTH + K - 1; j += TILE_WIDTH) {
                    int h_in = h_out + i - threadIdx.y;
                    int w_in = w_out + j - threadIdx.x;
                    if (h_in < Height && w_in < Width) {
                        shared_input[i * (TILE_WIDTH + K - 1) + j] = input[
                            n * Channel * Height * Width +
                            c * Height * Width +
                            h_in * Width +
                            w_in];
                    } else {
                        shared_input[i * (TILE_WIDTH + K - 1) + j] = 0.0f;
                    }
                }
            }
            __syncthreads();  // Ensure all data is loaded into shared memory

            // Perform convolution
            #pragma unroll  // Unroll the loop for better performance
            for (int p = 0; p < K; ++p) {
                #pragma unroll
                for (int q = 0; q < K; ++q) {
                    float mask_value = const_mask[
                        m * Channel * K * K +
                        c * K * K +
                        p * K +
                        q];
                    float input_value = shared_input[
                        (threadIdx.y + p) * (TILE_WIDTH + K - 1) +
                        threadIdx.x + q];
                    acc += input_value * mask_value;
                }
            }
            __syncthreads();  // Synchronize before loading the next channel
        }

        // Write the result to the output
        output[n * Map_out * Height_out * Width_out +
               m * Height_out * Width_out +
               h_out * Width_out +
               w_out] = acc;
    }
}

/**
 * @brief Prolog function to allocate device memory and copy data from host to device.
 */
__host__ void GPUInterface::conv_forward_gpu_prolog(
    const float *host_output, const float *host_input, const float *host_mask,
    float **device_output_ptr, float **device_input_ptr,
    const int Batch, const int Map_out, const int Channel,
    const int Height, const int Width, const int K) {

    // Calculate sizes
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);

    // Allocate device memory for input
    cudaMalloc((void**) device_input_ptr, input_size);
    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);

    // Allocate device memory for output
    cudaMalloc((void**) device_output_ptr, output_size);
    cudaMemset(*device_output_ptr, 0, output_size);  // Initialize output to zero

    // Copy mask to constant memory
    cudaMemcpyToSymbol(const_mask, host_mask, mask_size);

    // Error checking
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA error (prolog): " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

/**
 * @brief Main convolution function that launches the kernel.
 */
__host__ void GPUInterface::conv_forward_gpu(
    float *device_output, const float *device_input,
    const int Batch, const int Map_out, const int Channel,
    const int Height, const int Width, const int K) {

    // Output dimensions
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Block and grid dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((Height_out + TILE_WIDTH - 1) / TILE_WIDTH,  // Number of vertical tiles
                 Map_out,                                     // One block per output feature map
                 Batch);                                      // One block per image in the batch

    // Calculate shared memory size
    size_t shared_mem_size = (TILE_WIDTH + K - 1) * (TILE_WIDTH + K - 1) * sizeof(float);

    // Launch the convolution kernel
    conv_forward_kernel<<<dimGrid, dimBlock, shared_mem_size>>>(
        device_input,
        device_output,
        Batch,
        Map_out,
        Channel,
        Height,
        Width,
        K);

    // Error checking
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA error (conv_forward_kernel): " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

/**
 * @brief Epilog function to copy the output back to host and free device memory.
 */
__host__ void GPUInterface::conv_forward_gpu_epilog(
    float *host_output, float *device_output, float *device_input,
    const int Batch, const int Map_out, const int Channel,
    const int Height, const int Width, const int K) {

    // Calculate output size
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);

    // Copy the output back to host
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);

    // Error checking
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA error (epilog): " << cudaGetErrorString(error) << std::endl;
        exit(-1);
    }
}

/**
 * @brief Function to get and print the device properties.
 */
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
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, "
                  << deviceProp.maxThreadsDim[1] << " y, "
                  << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, "
                  << deviceProp.maxGridSize[1] << " y, "
                  << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}
