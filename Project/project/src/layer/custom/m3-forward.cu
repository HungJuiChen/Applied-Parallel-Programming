#include <cmath>
#include <iostream>
#include <cuda.h>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define MAX_BATCH_SIZE 1000

// im2col kernel: Unroll input into a 2D matrix
__global__ void im2col_kernel(const float *input, float *input_unrolled,
                              int Batch, int Channel, int Height, int Width, int K) {
    // Each thread maps to one element in input_unrolled
    // input_unrolled is of size (Channel*K*K) x (Batch*Height_out*Width_out)
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;

    int col = blockIdx.x * blockDim.x + threadIdx.x;  // index in W_unroll dimension
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // index in H_unroll dimension

    int H_unroll = Channel * K * K;
    int W_unroll = Batch * Height_out * Width_out;

    if (row < H_unroll && col < W_unroll) {
        int c = row / (K * K);
        int remainder = row % (K * K);
        int p = remainder / K;
        int q = remainder % K;

        int b = col / (Height_out * Width_out);
        int col_remainder = col % (Height_out * Width_out);
        int h = col_remainder / Width_out;
        int w = col_remainder % Width_out;

        int input_row = h + p;
        int input_col = w + q;

        if (b < Batch && c < Channel && input_row < Height && input_col < Width) {
            input_unrolled[row * W_unroll + col] = input[b * (Channel * Height * Width)
                                                        + c * (Height * Width)
                                                        + input_row * Width
                                                        + input_col];
        } else {
            input_unrolled[row * W_unroll + col] = 0.0f;
        }
    }
}

// Tiled GEMM kernel: C = A * B
// A is of dimension M x K (Map_out x (Channel*K*K))
// B is of dimension K x N ((Channel*K*K) x (Batch*Height_out*Width_out))
// C is of dimension M x N (Map_out x (Batch*Height_out*Width_out))
__global__ void gemm_kernel(const float *A, const float *B, float *C,
                            int M, int K, int N) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float val = 0.0f;

    // Loop over the tiles of K dimension
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        int tiled_k = t * TILE_WIDTH;

        if (row < M && (tiled_k + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + (tiled_k + threadIdx.x)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && (tiled_k + threadIdx.y) < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(tiled_k + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
            val += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = val;
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

}

__host__ void GPUInterface::conv_forward_gpu(float *host_output, const float *host_input, const float *host_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Derived dimensions
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int M = Map_out;
    int CKK = Channel * K * K; // This is the "K" dimension in GEMM sense
    int N = Batch * Height_out * Width_out;

    // Allocate device memory
    float *device_input, *device_mask, *device_output;
    float *device_input_unrolled;

    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);
    size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);

    cudaMalloc((void**)&device_input, input_size);
    cudaMemcpy(device_input, host_input, input_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&device_mask, mask_size);
    cudaMemcpy(device_mask, host_mask, mask_size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&device_output, output_size);

    // im2col matrix: dimensions (CKK x N)
    cudaMalloc((void**)&device_input_unrolled, CKK * N * sizeof(float));

    // Launch im2col kernel
    {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH,
                     (CKK + TILE_WIDTH - 1) / TILE_WIDTH);
        im2col_kernel<<<gridDim, blockDim>>>(device_input, device_input_unrolled,
                                             Batch, Channel, Height, Width, K);
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error (im2col): " << cudaGetErrorString(error) << std::endl;
            exit(-1);
        }
    }

    // Now perform GEMM: (M x CKK) * (CKK x N) = (M x N)
    {
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH,
                     (M + TILE_WIDTH - 1) / TILE_WIDTH);
        gemm_kernel<<<gridDim, blockDim>>>(device_mask, device_input_unrolled, device_output,
                                           M, CKK, N);
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error (gemm): " << cudaGetErrorString(error) << std::endl;
            exit(-1);
        }
    }

    // Copy result back
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(device_output);
    cudaFree(device_input_unrolled);

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cout<<"CUDA error (final): "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

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
