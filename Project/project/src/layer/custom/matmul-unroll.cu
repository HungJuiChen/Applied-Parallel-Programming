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
    const int H_unroll = Channel * K * K;
    const int W_unroll = Batch * Height_out * Width_out;

    //testing
    // const int total_elements = H_unroll * W_unroll;

    // int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate h_unroll and w_unroll using 2D grid and block indices
    int h_unroll = blockIdx.y * blockDim.y + threadIdx.y;
    int w_unroll = blockIdx.x * blockDim.x + threadIdx.x;

    // if (index < total_elements) {
    //     int h_unroll = index / W_unroll;
    //     int w_unroll = index % W_unroll;
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

        #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
        #define out_2d(i1, i0) output[(i1) * W_unroll + (i0)]
        // TODO: Insert your input matrix unrolling kernel code here
        // Implementing the input matrix unrolling as per instructions
        // Each thread handles one element in the unrolled output matrix

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

    // // Allocate device memory for input
    // size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    // cudaMalloc((void**) device_input_ptr, input_size);
    // cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);

    // // Allocate device memory for output
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    // size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);
    // cudaMalloc((void**) device_output_ptr, output_size);

    // Allocate device memory for mask
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);
    cudaMalloc((void**) device_mask_ptr, mask_size);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);

    // Set device_input_ptr and device_output_ptr to NULL as they are managed per chunk
    *device_input_ptr = NULL;
    *device_output_ptr = NULL;

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
    const int max_batch_size = 1000; // Adjust this value based on your GPU memory
    // const int Height_unrolled = Channel * K * K;
    // const int Width_unrolled = Batch * Height_out * Width_out;
    
    for (int b_start = 0; b_start < Batch; b_start += max_batch_size) {
        int current_batch_size = min(max_batch_size, Batch - b_start);

        size_t input_chunk_size = current_batch_size * Channel * Height * Width * sizeof(float);
        size_t output_chunk_size = current_batch_size * Map_out * Height_out * Width_out * sizeof(float);

        // Allocate device memory for input and output for the chunk
        float *device_input_chunk;
        float *device_output_chunk;
        cudaMalloc((void**)&device_input_chunk, input_chunk_size);
        cudaMalloc((void**)&device_output_chunk, output_chunk_size);

        // Copy input data for the chunk from host to device
        cudaMemcpy(device_input_chunk, device_input + b_start * Channel * Height * Width, input_chunk_size, cudaMemcpyHostToDevice);

        // Proceed with the computation for the current chunk
        const int Height_unrolled = Channel * K * K;
        const int Width_unrolled = current_batch_size * Height_out * Width_out;

        float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
        float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
        cudaMalloc((void**)&unrolled_matrix, Height_unrolled * Width_unrolled * sizeof(float));
        cudaMalloc((void**)&matmul_output, Map_out * Width_unrolled * sizeof(float));
        
        // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
        
        // int total_unrolled_elements = Height_unrolled * Width_unrolled;
        // int threads_per_block = BLOCK_SIZE;
        // int num_blocks = (total_unrolled_elements + threads_per_block - 1) / threads_per_block;

        // // Call the matrix unrolling kernel
        // matrix_unrolling_kernel<<<num_blocks, threads_per_block>>>(device_input, unrolled_matrix,
        //                                                            Batch, Channel, Height, Width, K);

        // Set the kernel dimensions for unrolling using a 2D grid
        dim3 blockDim_unroll(16, 16);
        dim3 gridDim_unroll((Width_unrolled + blockDim_unroll.x - 1) / blockDim_unroll.x,
                            (Height_unrolled + blockDim_unroll.y - 1) / blockDim_unroll.y);

        // Call the matrix unrolling kernel
        matrix_unrolling_kernel<<<gridDim_unroll, blockDim_unroll>>>(device_input, unrolled_matrix,
                                                                    Batch, Channel, Height, Width, K);

        // Check for errors
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            std::cout<<"CUDA error (unrolling kernel): "<<cudaGetErrorString(error)<<std::endl;
            exit(-1);
        }

        // TODO: Set the kernel dimensions and call the matmul kernel
        int numARows = Map_out;
        int numAColumns = Channel * K * K;
        int numBRows = Channel * K * K;
        int numBColumns = Batch * Height_out * Width_out;
        int numCRows = Map_out;
        int numCColumns = Batch * Height_out * Width_out;

        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((numCColumns - 1)/TILE_WIDTH + 1, (numCRows -1)/TILE_WIDTH + 1);

        // Call the matrix multiplication kernel
        matrixMultiplyShared<<<dimGrid, dimBlock>>>(device_mask, unrolled_matrix, matmul_output,
                                                    numARows, numAColumns,
                                                    numBRows, numBColumns,
                                                    numCRows, numCColumns);

        // Check for errors
        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            std::cout<<"CUDA error (matmul kernel): "<<cudaGetErrorString(error)<<std::endl;
            exit(-1);
        }

        // Permute the result of matrix multiplication
        const int out_image_size = Height_out * Width_out;
        dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
        matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
            matmul_output, device_output, Map_out, Batch, out_image_size
        );

        // Copy output data from device to host
        cudaMemcpy(device_output + b_start * Map_out * Height_out * Width_out,
                    device_output_chunk, output_chunk_size, cudaMemcpyDeviceToHost);

        cudaFree(matmul_output);
        cudaFree(unrolled_matrix);
        cudaFree(device_input_chunk);
        cudaFree(device_output_chunk);
    }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    // size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);
    // cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);
    
    // TODO: Free device memory

    // Free device memory
    // cudaFree(device_output);
    // cudaFree(device_input);
    cudaFree(device_mask);

    // Check for errors
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