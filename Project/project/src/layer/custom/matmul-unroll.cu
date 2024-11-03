#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

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

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define out_2d(i1, i0) output[(i1) * (Height_out * Width_out) + (i0)]
    // TODO: Insert your input matrix unrolling kernel code here
    // Implementing the input matrix unrolling as per instructions
    // Each thread handles one element in the unrolled output matrix

    int b = blockIdx.x; // Batch index
    int h_unroll = threadIdx.x; // Row index in the unrolled matrix

    if (h_unroll >= Channel * K * K) {
    printf("h_unroll out of bounds: %d >= %d\n", h_unroll, Channel * K * K);
    return;
    }

    if (h_unroll < Channel * K * K) {
        int c = h_unroll / (K * K);
        int p = (h_unroll % (K * K)) / K;
        int q = (h_unroll % (K * K)) % K;

        for (int h = 0; h < Height_out; ++h) {
            for (int w = 0; w < Width_out; ++w) {
                int w_unroll = h * Width_out + w;
                out_2d(h_unroll + b * (Channel * K * K), w_unroll) = in_4d(b, c, h + p, w + q);
            }
        }
    }


    #undef in_4d
    #undef out_2d
}

__host__ void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        exit(-1);
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

    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);
    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);

    cudaError_t err;

    // Allocate device memory and check for errors
    err = cudaMalloc((void**)device_input_ptr, input_size);
    checkCudaError(err, "Failed to allocate device memory for input");

    err = cudaMalloc((void**)device_mask_ptr, mask_size);
    checkCudaError(err, "Failed to allocate device memory for mask");

    err = cudaMalloc((void**)device_output_ptr, output_size);
    checkCudaError(err, "Failed to allocate device memory for output");

    // Copy data from host to device and check for errors
    err = cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy input data from host to device");

    err = cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy mask data from host to device");
    // Note: host_output does not need to be copied to device in prolog

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Batch * Height_out * Width_out;

    float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication

    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    
    cudaError_t err;

    err = cudaMalloc((void**)&unrolled_matrix, (Batch * Height_unrolled * Width_out) * sizeof(float));
    checkCudaError(err, "Failed to allocate device memory for unrolled matrix");

    err = cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));
    checkCudaError(err, "Failed to allocate device memory for matmul output");
    
    // Launch matrix unrolling kernel and check for errors
    dim3 unroll_grid_dim(Batch);
    dim3 unroll_block_dim(Height_unrolled);
    matrix_unrolling_kernel<<<unroll_grid_dim, unroll_block_dim>>>(device_input, unrolled_matrix, Batch, Channel, Height, Width, K);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    checkCudaError(err, "Kernel launch failure for matrix_unrolling_kernel");

    // Launch matrix multiplication kernel and check for errors
    dim3 matmul_grid_dim((Width_unrolled - 1) / TILE_WIDTH + 1, (Map_out - 1) / TILE_WIDTH + 1);
    dim3 matmul_block_dim(TILE_WIDTH, TILE_WIDTH);
    matrixMultiplyShared<<<matmul_grid_dim, matmul_block_dim>>>(device_mask, unrolled_matrix, matmul_output, 
                                                                Map_out, Channel * K * K, Channel * K * K, Width_unrolled, 
                                                                Map_out, Width_unrolled);
    err = cudaGetLastError();
    checkCudaError(err, "Kernel launch failure for matrixMultiplyShared");

    // TODO: Set the kernel dimensions and call the matmul kernel


    // Permute the result of matrix multiplication
    const int out_image_size = Height_out * Width_out;
    dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
        matmul_output, device_output, Map_out, Batch, out_image_size
    );
    err = cudaGetLastError();
    checkCudaError(err, "Kernel launch failure for matrix_permute_kernel");

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host

    size_t output_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float);

    // Copy output data from device to host and check for errors
    cudaError_t err = cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy output data from device to host");
    
    // TODO: Free device memory

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);

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