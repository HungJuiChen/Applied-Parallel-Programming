#include <cmath>
#include <iostream>
#include <cuda.h>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define MAX_BATCH_SIZE 1000  // Define a suitable maximum mini-batch size

// Define the CUDA error checking macro for cleaner code
#define CUDA_CHECK_ERROR(call)                                          \
    do {                                                                \
        cudaError_t error = call;                                       \
        if (error != cudaSuccess) {                                     \
            std::cerr << "CUDA Error: " << cudaGetErrorString(error)    \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while(0)

// Existing matrix_unrolling_kernel remains unchanged
__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
   
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

        // Compute linear indices
        int input_idx = b * (Channel * Height * Width) + c * (Height * Width) + input_row * Width + input_col;
        int output_idx = h_unroll * W_unroll + w_unroll;

        // Handle boundary conditions
        if (input_row < Height && input_col < Width) {
            output[output_idx] = input[input_idx];
        } else {
            output[output_idx] = 0.0f;
        }
    }
}

// Fused Matrix Multiplication and Permutation Kernel
__global__ void matrixMultiplyAndPermuteKernel(const float *A, const float *B, float *C,
                                               int numARows, int numAColumns,
                                               int numBRows, int numBColumns,
                                               int numCRows, int numCColumns,
                                               int Map_out, int Batch, int image_size) {
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float val = 0.0f;

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = A[row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = 0.0f;
        }

        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[(tileId * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            tileB[ty][tx] = 0.0f;
        }

        __syncthreads();

        if (row < numARows && col < numBColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }

        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        // Compute the batch index and map index based on the current thread
        int map = row; // row corresponds to Map_out
        int batch_position = col; // col corresponds to Batch * image_size

        int b = batch_position / image_size;
        int x = batch_position % image_size;

        // Compute the linear index for the output
        int output_idx = b * (Map_out * image_size) + map * image_size + x;

        // Write the result directly to the permuted output
        C[output_idx] = val;
    }
}

__host__ void GPUInterface::conv_forward_gpu(const float *host_input, const float *host_mask, float *host_output,
                                            const int Batch, const int Map_out, const int Channel,
                                            const int Height, const int Width, const int K)
{
    // CUDA event objects for timing (optional)
    cudaEvent_t start, stop;
    CUDA_CHECK_ERROR(cudaEventCreate(&start));
    CUDA_CHECK_ERROR(cudaEventCreate(&stop));

    CUDA_CHECK_ERROR(cudaEventRecord(start));

    // Calculate output dimensions
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int image_size = Height_out * Width_out;

    // Calculate unrolled dimensions
    const int Height_unrolled = Channel * K * K;

    // Determine the number of mini-batches
    int num_batches = (Batch + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;

    // Allocate device memory for input
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    float *device_input;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&device_input, input_size));

    // Allocate device memory for mask
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);
    float *device_mask;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&device_mask, mask_size));
    CUDA_CHECK_ERROR(cudaMemcpy(device_mask, host_mask, mask_size, cudaMemcpyHostToDevice));

    // Allocate device memory for output
    size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);
    float *device_output;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&device_output, output_size));
    // Initialize output to zero (optional, depending on the fusion logic)
    CUDA_CHECK_ERROR(cudaMemset(device_output, 0, output_size));

    // Allocate device memory for unrolled_matrix and matmul_output for the maximum mini-batch size
    size_t max_unroll_size = Height_unrolled * (MAX_BATCH_SIZE * Height_out * Width_out) * sizeof(float);
    float *unrolled_matrix;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&unrolled_matrix, max_unroll_size));

    // Copy the entire input in one go
    CUDA_CHECK_ERROR(cudaMemcpy(device_input, host_input, input_size, cudaMemcpyHostToDevice));

    // Iterate over each mini-batch
    for(int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        // Calculate the current mini-batch size
        int current_batch_size = (batch_idx == num_batches - 1) ? (Batch - batch_idx * MAX_BATCH_SIZE) : MAX_BATCH_SIZE;

        // Calculate current W_unroll
        int current_W_unroll = current_batch_size * Height_out * Width_out;

        // Set the kernel dimensions for unrolling using a 2D grid
        dim3 blockDim_unroll(16, 16);
        dim3 gridDim_unroll((current_W_unroll + blockDim_unroll.x - 1) / blockDim_unroll.x,
                            (Height_unrolled + blockDim_unroll.y - 1) / blockDim_unroll.y);
        
        // Call the matrix unrolling kernel for the current mini-batch
        matrix_unrolling_kernel<<<gridDim_unroll, blockDim_unroll>>>(
            device_input + batch_idx * MAX_BATCH_SIZE * Channel * Height * Width, // Offset input pointer
            unrolled_matrix, 
            current_batch_size, 
            Channel, 
            Height, 
            Width, 
            K
        );

        // Check for errors after unrolling kernel launch
        CUDA_CHECK_ERROR(cudaGetLastError());

        // Set the kernel dimensions for the fused matmul and permute kernel
        int numARows = Map_out;
        int numAColumns = Channel * K * K;
        int numBRows = Channel * K * K;
        int numBColumns = current_W_unroll;
        int numCRows = Map_out;
        int numCColumns = current_W_unroll;

        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((numCColumns + TILE_WIDTH - 1)/TILE_WIDTH, (numCRows + TILE_WIDTH -1)/TILE_WIDTH);

        // Call the fused matrix multiplication and permutation kernel
        matrixMultiplyAndPermuteKernel<<<dimGrid, dimBlock>>>(
            device_mask,          // A: [Map_out x (Channel * K * K)]
            unrolled_matrix,     // B: [(Channel * K * K) x (current_W_unroll)]
            device_output + batch_idx * MAX_BATCH_SIZE * Map_out * image_size, // C: [Batch x Map_out x image_size]
            numARows, numAColumns,
            numBRows, numBColumns,
            numCRows, numCColumns,
            Map_out, current_batch_size, image_size
        );

        // Check for errors after fused kernel launch
        CUDA_CHECK_ERROR(cudaGetLastError());
    }

    // Copy the output back to host
    CUDA_CHECK_ERROR(cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK_ERROR(cudaFree(device_input));
    CUDA_CHECK_ERROR(cudaFree(device_mask));
    CUDA_CHECK_ERROR(cudaFree(device_output));
    CUDA_CHECK_ERROR(cudaFree(unrolled_matrix));

    // Record and print elapsed time (optional)
    CUDA_CHECK_ERROR(cudaEventRecord(stop));
    CUDA_CHECK_ERROR(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "Convolution forward pass took " << milliseconds << " ms.\n";

    CUDA_CHECK_ERROR(cudaEventDestroy(start));
    CUDA_CHECK_ERROR(cudaEventDestroy(stop));
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    CUDA_CHECK_ERROR(cudaGetDeviceCount(&deviceCount));

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        CUDA_CHECK_ERROR(cudaGetDeviceProperties(&deviceProp, dev));

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
