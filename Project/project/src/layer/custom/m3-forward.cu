#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define MAX_BATCH_SIZE 5000

__global__ void fused_conv_kernel(const float *input, const float *mask, float *output,
                                  const int Batch, const int Map_out, const int Channel,
                                  const int Height, const int Width, const int K) {
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int H_unroll = Channel * K * K;
    const int W_unroll = Batch * Height_out * Width_out;

    // Shared memory for mask and input tiles
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y;  // Output feature map index (Map_out dimension)
    int bx = blockIdx.x;  // Column index in the unrolled input matrix
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty;  // Index in mask (filter weights)
    int col = bx * TILE_WIDTH + tx;  // Index in unrolled input

    float val = 0.0f;

    // Loop over tiles of the unrolled input
    for (int m = 0; m < (H_unroll - 1) / TILE_WIDTH + 1; ++m) {
        // Load mask tile into shared memory
        if (row < Map_out && m * TILE_WIDTH + tx < H_unroll) {
            tileA[ty][tx] = mask[row * H_unroll + m * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        // Compute indices for the input
        int h_unroll = m * TILE_WIDTH + ty;
        int w_unroll = col;

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

            if (b < Batch && c < Channel && input_row < Height && input_col < Width) {
                tileB[ty][tx] = input[b * (Channel * Height * Width) + c * (Height * Width) + input_row * Width + input_col];
            } else {
                tileB[ty][tx] = 0.0f;
            }
        } else {
            tileB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Perform the multiplication and accumulation
        if (row < Map_out && col < W_unroll) {
            #pragma unroll
            for (int k = 0; k < TILE_WIDTH; ++k) {
                val += tileA[ty][k] * tileB[k][tx];
            }
        }

        __syncthreads();
    }

    // Write the output directly to the correct position
    if (row < Map_out && col < W_unroll) {
        int b = col / (Height_out * Width_out);
        int remainder = col % (Height_out * Width_out);
        int h = remainder / Width_out;
        int w = remainder % Width_out;

        if (b < Batch && h < Height_out && w < Width_out) {
            output[b * (Map_out * Height_out * Width_out) + row * (Height_out * Width_out) + h * Width_out + w] = val;
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

    size_t current_W_unroll = Batch * Height_out * Width_out;


    // Set grid and block dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((current_W_unroll + TILE_WIDTH - 1) / TILE_WIDTH,
                    (Map_out + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch the fused kernel
    fused_conv_kernel<<<dimGrid, dimBlock>>>(
        device_input + batch_idx * Batch * Channel * Height * Width,
        device_mask,
        device_output + batch_idx * Batch * Map_out * Height_out * Width_out,
        Batch,
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

    // Determine the number of mini-batches
    // int num_batches = (Batch + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;

    // // Iterate over each mini-batch
    // for(int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    //     // Calculate the current mini-batch size
    //     int current_batch_size = (batch_idx == num_batches - 1) ? (Batch - batch_idx * MAX_BATCH_SIZE) : MAX_BATCH_SIZE;
    //     int current_W_unroll = current_batch_size * Height_out * Width_out;

    //     // Set grid and block dimensions
    //     dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    //     dim3 dimGrid((current_W_unroll + TILE_WIDTH - 1) / TILE_WIDTH,
    //                  (Map_out + TILE_WIDTH - 1) / TILE_WIDTH);

    //     // Launch the fused kernel
    //     fused_conv_kernel<<<dimGrid, dimBlock>>>(
    //         device_input + batch_idx * MAX_BATCH_SIZE * Channel * Height * Width,
    //         device_mask,
    //         device_output + batch_idx * MAX_BATCH_SIZE * Map_out * Height_out * Width_out,
    //         current_batch_size,
    //         Map_out,
    //         Channel,
    //         Height,
    //         Width,
    //         K
    //     );

    //     // Error checking
    //     cudaError_t error = cudaGetLastError();
    //     if(error != cudaSuccess) {
    //         std::cout << "CUDA error (fused kernel): " << cudaGetErrorString(error) << std::endl;
    //         exit(-1);
    //     }
    // }
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