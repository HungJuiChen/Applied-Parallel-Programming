#include <cmath>
#include <iostream>
#include <cuda.h>

#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define MAX_BATCH_SIZE 1000  // suitable maximum mini-batch size

// Unchanged kernel for matrix unrolling
__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
   
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    
    const int H_unroll = Channel * K * K;
    const int W_unroll = Batch * Height_out * Width_out;

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

        #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
        #define out_2d(i1, i0) output[(i1) * W_unroll + (i0)]

        if (input_row < Height && input_col < Width) {
            out_2d(h_unroll, w_unroll) = in_4d(b, c, input_row, input_col);
        } else {
            out_2d(h_unroll, w_unroll) = 0.0f;
        }

        #undef in_4d
        #undef out_2d
    }
}

// Fused kernel: Perform tiled matrix multiplication (Map_out x (C*K*K)) * ((C*K*K) x (Batch*H_out*W_out))
// and directly write the result into the permuted output layout: Batch x Map_out x H_out x W_out
__global__ void matrixMultiplyAndPermute(const float *A, const float *B, float *output,
                                         int Map_out, int Channel, int Height_out, int Width_out,
                                         int current_batch_size, int K)
{
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    int numARows = Map_out;
    int numAColumns = Channel * K * K;
    int numBRows = numAColumns;
    int numBColumns = current_batch_size * Height_out * Width_out;
    int numCRows = numARows;
    int numCColumns = numBColumns;

    int by = blockIdx.y, bx = blockIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;

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

        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val += tileA[ty][i] * tileB[i][tx];
            }
        }

        __syncthreads();
    }

    // Now write 'val' directly into the output in permuted order:
    // We have computed C[row, col] where:
    //   row in [0, Map_out)
    //   col in [0, current_batch_size * Height_out * Width_out)
    // Permute into output[b, row, h, w]
    if (row < numCRows && col < numCColumns) {
        int b = col / (Height_out * Width_out);
        int remainder = col % (Height_out * Width_out);
        int h = remainder / Width_out;
        int w = remainder % Width_out;
        // output shape: Batch x Map_out x Height_out x Width_out
        output[b * (Map_out * Height_out * Width_out) + row * (Height_out * Width_out) + h * Width_out + w] = val;
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

}

__host__ void GPUInterface::conv_forward_gpu(float *host_output, const float *host_input, const float *host_mask,
                        const int Batch, const int Map_out, const int Channel,
                        const int Height, const int Width, const int K) {

    // Allocate device memory for input, output, mask
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    float *device_input, *device_mask, *device_output;
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);
    size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);

    cudaMalloc(&device_input, input_size);
    cudaMalloc(&device_output, output_size);
    cudaMalloc(&device_mask, mask_size);

    cudaMemcpy(device_input, host_input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_mask, host_mask, mask_size, cudaMemcpyHostToDevice);

    // Determine number of mini-batches
    int num_batches = (Batch + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;

    // Allocate device memory for intermediate unrolled matrix
    // Max sizes
    int Height_unrolled = Channel * K * K;
    size_t max_unroll_size = Height_unrolled * (MAX_BATCH_SIZE * Height_out * Width_out) * sizeof(float);
    float *unrolled_matrix;
    cudaMalloc(&unrolled_matrix, max_unroll_size);

    // For each mini-batch
    for(int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        int current_batch_size = (batch_idx == num_batches - 1) ? (Batch - batch_idx * MAX_BATCH_SIZE) : MAX_BATCH_SIZE;
        int current_W_unroll = current_batch_size * Height_out * Width_out;

        // Launch unrolling kernel
        dim3 blockDim_unroll(16, 16);
        dim3 gridDim_unroll((current_W_unroll + blockDim_unroll.x - 1) / blockDim_unroll.x,
                            (Height_unrolled + blockDim_unroll.y - 1) / blockDim_unroll.y);

        matrix_unrolling_kernel<<<gridDim_unroll, blockDim_unroll>>>(
            device_input + batch_idx * MAX_BATCH_SIZE * Channel * Height * Width,
            unrolled_matrix,
            current_batch_size,
            Channel,
            Height,
            Width,
            K
        );
        cudaDeviceSynchronize();

        // Matrix multiply and permute
        int numCColumns = current_W_unroll;
        int numCRows = Map_out;

        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((numCColumns - 1)/TILE_WIDTH + 1, (numCRows -1)/TILE_WIDTH + 1);

        matrixMultiplyAndPermute<<<dimGrid, dimBlock>>>(
            device_mask,
            unrolled_matrix,
            device_output + batch_idx * MAX_BATCH_SIZE * Map_out * Height_out * Width_out,
            Map_out,
            Channel,
            Height_out,
            Width_out,
            current_batch_size,
            K
        );
        cudaDeviceSynchronize();
    }

    // Copy result back to host
    cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
    cudaFree(unrolled_matrix);
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

