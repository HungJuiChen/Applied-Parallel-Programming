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
__global__ void scan(float *input, float *output, float *blockSums, int len) {
  __shared__ float temp[2 * BLOCK_SIZE]; // allocate shared memory

  int t = threadIdx.x;
  int start = 2 * blockIdx.x * blockDim.x;
  int idx = start + t;

  // Load input elements into shared memory
  if (idx < len) {
    temp[t] = input[idx];
  } else {
    temp[t] = 0.0f;
  }
  if (idx + blockDim.x < len) {
    temp[t + blockDim.x] = input[idx + blockDim.x];
  } else {
    temp[t + blockDim.x] = 0.0f;
  }

  int n = 2 * blockDim.x;
  int offset = 1;

  // Up-sweep phase
  for (int d = n >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if (t < d) {
      int ai = offset * (2 * t + 1) - 1;
      int bi = offset * (2 * t + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset <<= 1;
  }

  // Store block sum and clear last element
  if (t == 0) {
    if (blockSums != NULL) {
      blockSums[blockIdx.x] = temp[n - 1];
    }
    temp[n - 1] = 0;
  }

  // Down-sweep phase
  for (int d = 1; d < n; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    if (t < d) {
      int ai = offset * (2 * t + 1) - 1;
      int bi = offset * (2 * t + 2) - 1;
      float temp_ai = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += temp_ai;
    }
  }
  __syncthreads();

  // Write results to device memory
  if (idx < len) {
    output[idx] = temp[t];
  }
  if (idx + blockDim.x < len) {
    output[idx + blockDim.x] = temp[t + blockDim.x];
  }
}


__global__ void addScannedBlockSums(float *output, float *scannedBlockSums, int len) {
  int t = threadIdx.x;
  int start = 2 * blockIdx.x * blockDim.x;
  int idx = start + t;
  float addValue = 0.0f;
  if (blockIdx.x > 0) {
    addValue = scannedBlockSums[blockIdx.x - 1];
  }
  if (idx < len) {
    output[idx] += addValue;
  }
  if (idx + blockDim.x < len) {
    output[idx + blockDim.x] += addValue;
  }
}

unsigned int nextPowerOfTwo(unsigned int x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return x;
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

  // Allocate memory for block sums
  int numBlocks = (numElements + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
  wbCheck(cudaMalloc((void **)&deviceBlockSums, numBlocks * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScannedBlockSums, numBlocks * sizeof(float)));

  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  // Initialize the grid and block dimensions here
  dim3 blockDim(BLOCK_SIZE, 1, 1);
  dim3 gridDim(numBlocks, 1, 1);

  // First pass: scan per block and compute block sums
  scan<<<gridDim, blockDim>>>(deviceInput, deviceOutput, deviceBlockSums, numElements);
  cudaDeviceSynchronize();

  // If we have multiple blocks, scan the block sums
  if (numBlocks > 1) {
    unsigned int nBlocks2 = nextPowerOfTwo(numBlocks);
    dim3 blockDim2(nBlocks2 / 2, 1, 1);
    dim3 gridDim2(1, 1, 1);

    // Second pass: scan the block sums
    scan<<<gridDim2, blockDim2>>>(deviceBlockSums, deviceScannedBlockSums, NULL, numBlocks);
    cudaDeviceSynchronize();

    // Third pass: add scanned block sums to each block
    addScannedBlockSums<<<gridDim, blockDim>>>(deviceOutput, deviceScannedBlockSums, numElements);
    cudaDeviceSynchronize();
  }

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));

  // Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceBlockSums);
  cudaFree(deviceScannedBlockSums);

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

