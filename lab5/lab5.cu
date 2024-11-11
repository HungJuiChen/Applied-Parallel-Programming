// MP5 Reduction
// Input: A num list of length n
// Output: Sum of the list = list[0] + list[1] + ... + list[n-1];

#include <wb.h>

#define BLOCK_SIZE 512 //@@ This value is not fixed and you can adjust it according to the situation

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
  
__global__ void total(float *input, float *output, int len) {
  __shared__ float partialSum[BLOCK_SIZE];  // Allocate shared memory

  int tid = threadIdx.x;
  int globalIndex = blockIdx.x * (BLOCK_SIZE * 2) + threadIdx.x;

  // Load elements from the input array into shared memory, considering boundary checks
  if (globalIndex < len) {
      partialSum[tid] = input[globalIndex] + ((globalIndex + BLOCK_SIZE) < len ? input[globalIndex + BLOCK_SIZE] : 0.0f);
  } else {
      partialSum[tid] = 0.0f;
  }

  __syncthreads();  

  // Reduction in shared memory
  for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
          partialSum[tid] += partialSum[tid + stride];
      }
      __syncthreads(); 
  }

  // Write the result of this block's reduction to the output array
  if (tid == 0) {
      output[blockIdx.x] = partialSum[0];
  }
}

int main(int argc, char **argv) {
  int ii;
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  //@@ Initialize device input and output pointers
  float *deviceInput;
  float *deviceOutput;

  int numInputElements;  // number of elements in the input list
  int numOutputElements; // number of elements in the output list

  args = wbArg_read(argc, argv);

  //Import data and create memory on host
  hostInput =
      (float *)wbImport(wbArg_getInputFile(args, 0), &numInputElements);

  numOutputElements = numInputElements / (BLOCK_SIZE << 1);
  if (numInputElements % (BLOCK_SIZE << 1)) {
    numOutputElements++;
  }
  hostOutput = (float *)malloc(numOutputElements * sizeof(float));

  // The number of input elements in the input is numInputElements
  // The number of output elements in the input is numOutputElements

  //@@ Allocate GPU memory
  wbCheck(cudaMalloc((void **)&deviceInput, numInputElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numOutputElements * sizeof(float)));


  //@@ Copy input memory to the GPU
  wbCheck(cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice));


  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimGrid((numInputElements + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2), 1, 1);


  //@@ Launch the GPU Kernel and perform CUDA computation
  total<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numInputElements);
  
  cudaDeviceSynchronize();  
  //@@ Copy the GPU output memory back to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost));
  
  /********************************************************************
   * Reduce output vector on the host
   * NOTE: One could also perform the reduction of the output vector
   * recursively and support any size input. 
   * For simplicity, we do not require that for this lab.
   ********************************************************************/
  for (ii = 1; ii < numOutputElements; ii++) {
    hostOutput[0] += hostOutput[ii];
  }

  //@@ Free the GPU memory
  wbCheck(cudaFree(deviceInput));
  wbCheck(cudaFree(deviceOutput));


  wbSolution(args, hostOutput, 1);

  free(hostInput);
  free(hostOutput);

  return 0;
}

