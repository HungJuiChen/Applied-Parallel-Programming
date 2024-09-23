#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define TILE_WIDTH 16

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
    // Shared memory for tiles of A and B
  __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x;  // Block index (x dimension)
  int by = blockIdx.y;  // Block index (y dimension)
  int tx = threadIdx.x; // Thread index (x dimension)
  int ty = threadIdx.y; // Thread index (y dimension)

  // Row and column index of the element to be computed
  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;

  float value = 0.0;

  // Loop over all the tiles of A and B required to compute the C element
  for (int t = 0; t < (numAColumns - 1) / TILE_WIDTH + 1; ++t) {
    // Load A and B tiles into shared memory
    if (row < numARows && t * TILE_WIDTH + tx < numAColumns) {
      sharedA[ty][tx] = A[row * numAColumns + t * TILE_WIDTH + tx];
    } else {
      sharedA[ty][tx] = 0.0;
    }

    if (t * TILE_WIDTH + ty < numBRows && col < numBColumns) {
      sharedB[ty][tx] = B[(t * TILE_WIDTH + ty) * numBColumns + col];
    } else {
      sharedB[ty][tx] = 0.0;
    }

    __syncthreads(); // Synchronize threads to make sure tiles are loaded

    // Perform the multiplication for this tile
    for (int k = 0; k < TILE_WIDTH; ++k) {
      value += sharedA[ty][k] * sharedB[k][tx];
    }

    __syncthreads(); // Synchronize threads before loading new tiles
  }

  // Write the computed value to C
  if (row < numCRows && col < numCColumns) {
    C[row * numCColumns + col] = value;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix

  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  hostC = (float *)malloc(numCRows * numCColumns * sizeof(float));  

  //@@ Allocate GPU memory here
  float *deviceA, *deviceB, *deviceC;
  wbCheck(cudaMalloc((void **)&deviceA, numARows * numAColumns * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceB, numBRows * numBColumns * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceC, numCRows * numCColumns * sizeof(float)));

  //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((numCColumns - 1) / TILE_WIDTH + 1, (numCRows - 1) / TILE_WIDTH + 1, 1);  

  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC,
                                              numARows, numAColumns,
                                              numBRows, numBColumns,
                                              numCRows, numCColumns);
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost)); 

  //@@ Free the GPU memory here
  wbCheck(cudaFree(deviceA));
  wbCheck(cudaFree(deviceB));
  wbCheck(cudaFree(deviceC));

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);

  //@@ Free the hostC matrix
  free(hostC);

  return 0;
}
