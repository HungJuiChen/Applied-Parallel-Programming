#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"


#define TILE_WIDTH 16
#define BLOCK_SIZE 256
#define MAX_BATCH_SIZE 1000

// Global variables to store host pointers
const float *global_host_input;
const float *global_host_output;
const float *global_host_mask;

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

    // Allocate pinned host memory for input and mask
    // size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    // cudaMallocHost((void**)&host_input_pinned, input_size);
    // memcpy(host_input_pinned, host_input, input_size);

    // size_t mask_size = Map_out * Channel * K * K * sizeof(float);
    // cudaMallocHost((void**)&host_mask_pinned, mask_size);
    // memcpy(host_mask_pinned, host_mask, mask_size);

    // // Allocate device memory for input
    // cudaMalloc((void**) device_input_ptr, input_size);
    // // Asynchronously copy input to device using stream 0
    // cudaMemcpyAsync(*device_input_ptr, host_input_pinned, input_size, cudaMemcpyHostToDevice, streams[0]);

    // // Allocate device memory for output
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    // size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);
    // cudaMalloc((void**) device_output_ptr, output_size);

    // // Allocate device memory for mask
    // cudaMalloc((void**) device_mask_ptr, mask_size);
    // // Asynchronously copy mask to device using stream 0
    // cudaMemcpyAsync(*device_mask_ptr, host_mask_pinned, mask_size, cudaMemcpyHostToDevice, streams[0]);

    // // Initialize CUDA Streams
    // for(int i = 0; i < NUM_STREAMS; ++i){
    //     cudaStreamCreate(&streams[i]);
    // }

    // Store host pointers in global variables
    global_host_input = host_input;
    global_host_output = host_output;  // Now non-const
    global_host_mask = host_mask;

    // Allocate device memory for mask (since it doesn't change, we can do this once)
    size_t mask_size = Map_out * Channel * K * K * sizeof(float);
    cudaMalloc((void**) device_mask_ptr, mask_size);
    cudaMemcpy(*device_mask_ptr, host_mask, mask_size, cudaMemcpyHostToDevice);

    // Allocate device memory for input
    size_t input_size = Batch * Channel * Height * Width * sizeof(float);
    cudaMalloc((void**) device_input_ptr, input_size);
    cudaMemcpy(*device_input_ptr, host_input, input_size, cudaMemcpyHostToDevice);

    // Allocate device memory for output
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);
    cudaMalloc((void**) device_output_ptr, output_size);

    // Check for errors
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cout<<"CUDA error (prolog): "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    //const int Width_unrolled = Batch * Height_out * Width_out;

    // Determine the number of mini-batches
    int num_batches = (Batch + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;

    float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    
    //cudaMalloc((void**)&unrolled_matrix, (size_t) Batch * Channel * K * K * Height_out * Width_out * sizeof(float));
    //cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));
    
    // Allocate device memory for unrolled_matrix and matmul_output for the maximum mini-batch size
    size_t max_unroll_size = Height_unrolled * (MAX_BATCH_SIZE * Height_out * Width_out) * sizeof(float);
    cudaMalloc((void**)&unrolled_matrix, max_unroll_size);

    size_t max_matmul_size = Map_out * (MAX_BATCH_SIZE * Height_out * Width_out) * sizeof(float);
    cudaMalloc((void**)&matmul_output, max_matmul_size);

    // Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // TODO: Set the kernel dimensions and call the matrix unrolling kernel.
    
    // Iterate over each mini-batch
    for(int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {

        // Determine the current stream
        //int stream_id = batch_idx % NUM_STREAMS;
        //cudaStream_t stream = streams[stream_id];

        // Calculate the current mini-batch size
        int current_batch_size = (batch_idx == num_batches - 1) ? (Batch - batch_idx * MAX_BATCH_SIZE) : MAX_BATCH_SIZE;

        // Calculate current W_unroll
        int current_W_unroll = current_batch_size * Height_out * Width_out;
        // Set the kernel dimensions for unrolling using a 2D grid
        
        // Allocate device memory for input and output of current batch
        float *device_input_batch;
        float *device_output_batch;
        size_t input_batch_size = current_batch_size * Channel * Height * Width * sizeof(float);
        size_t output_batch_size = current_batch_size * Map_out * Height_out * Width_out * sizeof(float);

        cudaMallocAsync((void**)&device_input_batch, input_batch_size, stream1);
        cudaMallocAsync((void**)&device_output_batch, output_batch_size, stream1);

        // Asynchronously copy input data of current batch
        cudaMemcpyAsync(device_input_batch,
                        global_host_input + batch_idx * MAX_BATCH_SIZE * Channel * Height * Width,
                        input_batch_size, cudaMemcpyHostToDevice, stream1);
        
        dim3 blockDim_unroll(16, 16);
        dim3 gridDim_unroll((current_W_unroll + blockDim_unroll.x - 1) / blockDim_unroll.x,
                            (Height_unrolled + blockDim_unroll.y - 1) / blockDim_unroll.y);
        
        // Synchronize stream1 to ensure data is copied before kernel execution
        cudaStreamSynchronize(stream1);

        // Call the matrix unrolling kernel for the current mini-batch
        // matrix_unrolling_kernel<<<gridDim_unroll, blockDim_unroll>>>(
        //     device_input + batch_idx * MAX_BATCH_SIZE * Channel * Height * Width, // Offset input pointer
        //     unrolled_matrix, 
        //     current_batch_size, 
        //     Channel, 
        //     Height, 
        //     Width, 
        //     K
        // );

        // Asynchronously call the matrix unrolling kernel on the current stream
        // matrix_unrolling_kernel<<<gridDim_unroll, blockDim_unroll, 0, stream>>>(
        //     device_input + batch_idx * MAX_BATCH_SIZE * Channel * Height * Width, // Offset input pointer
        //     unrolled_matrix, 
        //     current_batch_size, 
        //     Channel, 
        //     Height, 
        //     Width, 
        //     K
        // );

        // Launch the unrolling kernel
        matrix_unrolling_kernel<<<gridDim_unroll, blockDim_unroll, 0, stream2>>>(
            device_input_batch, unrolled_matrix, current_batch_size, Channel, Height, Width, K);

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
        int numBColumns = current_W_unroll;
        int numCRows = Map_out;
        int numCColumns = current_W_unroll;

        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 dimGrid((numCColumns - 1)/TILE_WIDTH + 1, (numCRows -1)/TILE_WIDTH + 1);

        // // Call the matrix multiplication kernel
        // matrixMultiplyShared<<<dimGrid, dimBlock>>>(device_mask, unrolled_matrix, matmul_output,
        //                                             numARows, numAColumns,
        //                                             numBRows, numBColumns,
        //                                             numCRows, numCColumns);

        // Asynchronously call the matrix multiplication kernel on the current stream
        // matrixMultiplyShared<<<dimGrid, dimBlock, 0, stream>>>(
        //     device_mask, unrolled_matrix, matmul_output,
        //     numARows, numAColumns,
        //     numBRows, numBColumns,
        //     numCRows, numCColumns
        // );

        // Launch the matrix multiplication kernel
        matrixMultiplyShared<<<dimGrid, dimBlock, 0, stream2>>>(
            device_mask, unrolled_matrix, matmul_output,
            numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            std::cout<<"CUDA error (matmul kernel): "<<cudaGetErrorString(error)<<std::endl;
            exit(-1);
        }

        // Permute the result of matrix multiplication
        const int out_image_size = Height_out * Width_out;
        dim3 permute_kernel_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, current_batch_size, 1);
        // matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
        //     matmul_output, device_output, Map_out, Batch, out_image_size
        // );
        // matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE>>>(
        //     matmul_output, 
        //     device_output + batch_idx * MAX_BATCH_SIZE * Map_out * out_image_size, // Offset output pointer
        //     Map_out, 
        //     current_batch_size, 
        //     out_image_size
        // );

        // Asynchronously call the permutation kernel on the current stream
        // matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE, 0, stream>>>(
        //     matmul_output, 
        //     device_output + batch_idx * MAX_BATCH_SIZE * Map_out * out_image_size, // Offset output pointer
        //     Map_out, 
        //     current_batch_size, 
        //     out_image_size
        // );

        matrix_permute_kernel<<<permute_kernel_grid_dim, BLOCK_SIZE, 0, stream2>>>(
            matmul_output, device_output_batch, Map_out, current_batch_size, out_image_size);

        // Check for errors after permutation
        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            std::cout<<"CUDA error (permute kernel): "<<cudaGetErrorString(error)<<std::endl;
            exit(-1);
        }

        // Asynchronously copy output data back to host
        cudaMemcpyAsync(global_host_output + batch_idx * MAX_BATCH_SIZE * Map_out * out_image_size,
                        device_output_batch, output_batch_size, cudaMemcpyDeviceToHost, stream2);

        // Free device memory of current batch
        cudaFreeAsync(device_input_batch, stream2);
        cudaFreeAsync(device_output_batch, stream2);
    }

    // Synchronize streams before cleanup
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // Destroy streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    // Free allocated memory
    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    // const int Height_out = Height - K + 1;
    // const int Width_out = Width - K + 1;
    // size_t output_size = Batch * Map_out * Height_out * Width_out * sizeof(float);
    // //cudaMemcpy(host_output, device_output, output_size, cudaMemcpyDeviceToHost);
    
    // // Asynchronously copy output from device to host using stream 0
    // cudaMemcpyAsync(host_output, device_output, output_size, cudaMemcpyDeviceToHost, streams[0]);

    // // Synchronize all streams to ensure copy is complete
    // for(int i = 0; i < NUM_STREAMS; ++i){
    //     cudaStreamSynchronize(streams[i]);
    // }

    // // TODO: Free device memory
    // cudaFree(device_output);
    // cudaFree(device_input);
    // cudaFree(device_mask);

    // // Free pinned host memory
    // cudaFreeHost(host_input_pinned);
    // cudaFreeHost(host_mask_pinned);
    // cudaFreeHost(host_output_pinned);

    // // Destroy CUDA Streams
    // for(int i = 0; i < NUM_STREAMS; ++i){
    //     cudaStreamDestroy(streams[i]);
    // }

    // Free device memory for mask (input and output are freed in conv_forward_gpu)
    cudaFree(device_mask);

    // Reset global pointers
    global_host_input = nullptr;
    global_host_output = nullptr;
    global_host_mask = nullptr;

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