#include <cmath>
#include <iostream>
#include <cuda.h>
#include <mma.h>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define MAX_BATCH_SIZE 1000

// Use the nvcuda::wmma namespace
using namespace nvcuda;

__global__ void tensor_conv_kernel(const float *input, const float *mask, float *output,
                                   const int Batch, const int Map_out, const int Channel,
                                   const int Height, const int Width, const int K) {
    
    extern __shared__ char shared_memory[];
    
    // Constants for the convolution
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // WMMA tile dimensions
    const int M = 16;  // Rows of output tile
    const int N = 16;  // Columns of output tile
    const int K_TILE = 16;  // Shared dimension

    // Calculate indices
    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    // Each warp handles one output tile
    int warpM = (tile_row * blockDim.y + threadIdx.y) / warpSize;
    int warpN = (tile_col * blockDim.x + threadIdx.x) / warpSize;

    // Initialize accumulator fragment
    wmma::fragment<wmma::accumulator, M, N, K_TILE, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Loop over K dimension
    for (int k = 0; k < Channel * K * K; k += K_TILE) {
        // Declare fragments
        wmma::fragment<wmma::matrix_a, M, K_TILE, K_TILE, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, K_TILE, N, K_TILE, half, wmma::col_major> b_frag;

        // Load mask fragment
        int a_row = warpM * M;
        int a_col = k;
        if (a_row < Map_out && a_col < Channel * K * K) {
            const half* a_ptr = reinterpret_cast<const half*>(mask + a_row * Channel * K * K + a_col);
            wmma::load_matrix_sync(a_frag, a_ptr, Channel * K * K);
        } else {
            wmma::fill_fragment(a_frag, __float2half(0.0f));
        }

        // Load input fragment
        int b_row = k;
        int b_col = warpN * N;
        if (b_row < Channel * K * K && b_col < Batch * Height_out * Width_out) {
            // Compute unrolled input indices on-the-fly
            //half* shared_B = reinterpret_cast<half*>(__shared__ + threadIdx.y * blockDim.x + threadIdx.x);
            half* shared_B = reinterpret_cast<half*>(shared_memory + (threadIdx.y * blockDim.x + threadIdx.x) * sizeof(half));
            int c = (b_row) / (K * K);
            int p = ((b_row) % (K * K)) / K;
            int q = ((b_row) % (K * K)) % K;

            int b = b_col / (Height_out * Width_out);
            int h = (b_col % (Height_out * Width_out)) / Width_out;
            int w = (b_col % (Height_out * Width_out)) % Width_out;

            int input_h = h + p;
            int input_w = w + q;

            if (b < Batch && c < Channel && input_h < Height && input_w < Width) {
                float val = input[b * (Channel * Height * Width) + c * (Height * Width) + input_h * Width + input_w];
                shared_B[threadIdx.y * blockDim.x + threadIdx.x] = __float2half(val);
            } else {
                shared_B[threadIdx.y * blockDim.x + threadIdx.x] = __float2half(0.0f);
            }

            wmma::load_matrix_sync(b_frag, shared_B, Batch * Height_out * Width_out);
        } else {
            wmma::fill_fragment(b_frag, __float2half(0.0f));
        }

        // Perform matrix multiplication
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    // Store the accumulator fragment to global memory
    int c_row = warpM * M;
    int c_col = warpN * N;
    if (c_row < Map_out && c_col < Batch * Height_out * Width_out) {
        float* c_ptr = output + c_row * Batch * Height_out * Width_out + c_col;
        wmma::store_matrix_sync(c_ptr, acc, Batch * Height_out * Width_out, wmma::mem_row_major);
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

    // WMMA tile dimensions
    const int M = 16;
    const int N = 16;
    const int K_TILE = 16;

    // Determine grid and block dimensions
    dim3 dimBlock(32, 8);  // 256 threads per block
    dim3 dimGrid((Batch * Height_out * Width_out + N - 1) / N,
                 (Map_out + M - 1) / M);

    // Launch the kernel
    tensor_conv_kernel<<<dimGrid, dimBlock>>>(
        device_input,
        device_mask,
        device_output,
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
        std::cout << "CUDA error (tensor kernel): " << cudaGetErrorString(error) << std::endl;
        exit(-1);
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