// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Implement the work-efficient scan kernel to generate per-block scan array and store the block sums into an auxiliary block sum array.
// Use shared memory to reduce the number of global memory accesses, handle the boundary conditions when loading input list elements into the shared memory
__global__ void scan_block_kernel(float *input, float *output, float *block_sums, int len) {
  __shared__ float shared_data[BLOCK_SIZE];
  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x + tid;

  // Load data into shared memory
  if (index < len) {
    shared_data[tid] = input[index];
  } else {
    shared_data[tid] = 0;
  }
  __syncthreads();

  // Up-sweep (reduction) phase
  for (int stride = 1; stride <= tid; stride <<= 1) {
    int idx = (tid - stride);
    if (idx >= 0)
      shared_data[tid] += shared_data[idx];
    __syncthreads();
  }

  // Store the last element of each block into block_sums
  if (tid == BLOCK_SIZE - 1 || index == len - 1) {
    block_sums[blockIdx.x] = shared_data[tid];
  }

  // Write the results to output
  if (index < len) {
    output[index] = shared_data[tid];
  }
}

// Implement the kernel that adds the accumulative block sums to the appropriate elements of the per-block scan array to complete the scan for all the elements.
__global__ void add_block_sums(float *output, float *block_sums, int len) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (blockIdx.x > 0 && index < len) {
    output[index] += block_sums[blockIdx.x - 1];
  }
}

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float shared_data[BLOCK_SIZE];
  int tid = threadIdx.x;

  // Load data into shared memory
  if (tid < len) {
    shared_data[tid] = input[tid];
  } else {
    shared_data[tid] = 0;
  }
  __syncthreads();

  // Up-sweep (reduction) phase
  for (int stride = 1; stride <= tid; stride <<= 1) {
    int idx = tid - stride;
    if (idx >= 0)
      shared_data[tid] += shared_data[idx];
    __syncthreads();
  }

  // Write the results to output
  if (tid < len) {
    output[tid] = shared_data[tid];
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceBlockSums;
  float *deviceScannedBlockSums;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));


  // Allocate GPU memory.
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));


  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  int numBlocks = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
  dim3 blockDim(BLOCK_SIZE);
  dim3 gridDim(numBlocks);

  // Allocate memory for block sums
  wbCheck(cudaMalloc((void **)&deviceBlockSums, numBlocks * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScannedBlockSums, numBlocks * sizeof(float)));

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce

  // First, perform scan on each block
  scan_block_kernel<<<gridDim, blockDim>>>(deviceInput, deviceOutput, deviceBlockSums, numElements);
  cudaDeviceSynchronize();

  // Then, perform scan on block sums array
  scan<<<1, BLOCK_SIZE>>>(deviceBlockSums, deviceScannedBlockSums, numBlocks);
  cudaDeviceSynchronize();
  
  // Finally, add the scanned block sums to the per-block scan results
  add_block_sums<<<gridDim, blockDim>>>(deviceOutput, deviceScannedBlockSums, numElements);
  cudaDeviceSynchronize();

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));


  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceBlockSums);
  cudaFree(deviceScannedBlockSums);

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}