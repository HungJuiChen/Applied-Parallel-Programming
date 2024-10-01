#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define BLOCK_SIZE 8  // Define the size of CUDA blocks (8x8x8)

//@@ Define constant memory for device kernel here
__constant__ float d_Kernel[27];  // 3x3x3 convolution kernel stored in constant memory

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  // Calculate the 3D index of the current thread
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  // Calculate the linear index for the output
  int output_index = z * y_size * x_size + y * x_size + x;

  if (x < x_size && y < y_size && z < z_size) {
    float sum = 0.0f;

    // Iterate over the 3x3x3 kernel
    for (int kz = -1; kz <= 1; kz++) {
      for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
          int in_z = z + kz;
          int in_y = y + ky;
          int in_x = x + kx;

          // Check for boundary conditions
          if (in_z >= 0 && in_z < z_size &&
              in_y >= 0 && in_y < y_size &&
              in_x >= 0 && in_x < x_size) {
            int input_index = in_z * y_size * x_size + in_y * x_size + in_x;
            float input_val = input[input_index];
            float kernel_val = d_Kernel[(kz + 1) * 9 + (ky + 1) * 3 + (kx + 1)];
            sum += input_val * kernel_val;
          }
        }
      }
    }

    output[output_index] = sum;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  wbCheck(cudaMalloc((void **)&deviceInput, (inputLength - 3) * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, (inputLength - 3) * sizeof(float)));


  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  wbCheck(cudaMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice));
  // Copy the kernel to constant memory
  wbCheck(cudaMemcpyToSymbol(d_Kernel, hostKernel, kernelLength * sizeof(float)));


  //@@ Initialize grid and block dimensions here
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(
      (x_size + blockDim.x - 1) / blockDim.x,
      (y_size + blockDim.y - 1) / blockDim.y,
      (z_size + blockDim.z - 1) / blockDim.z
  );
  //@@ Launch the GPU kernel here
  conv3d<<<gridDim, blockDim>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  wbCheck(cudaGetLastError());

  // Wait for the GPU to finish
  cudaDeviceSynchronize();



  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  wbCheck(cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost));



  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  wbCheck(cudaFree(deviceInput));
  wbCheck(cudaFree(deviceOutput));

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

