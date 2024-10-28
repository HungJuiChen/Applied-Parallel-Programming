// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here

// Define the wbCheck macro for CUDA error checking
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Kernel to cast float image data [0,1] to unsigned char [0,255]
__global__ void castToUChar(const float *input, unsigned char *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (unsigned char)(input[idx] * 255.0f);
    }
}

// Kernel to convert RGB image to Grayscale using the luminosity method
__global__ void rgbToGrayscale(const unsigned char *rgbImage, unsigned char *grayImage, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int numPixels = width * height;
    if (idx < numPixels) {
        int rgbIdx = 3 * idx; // Assuming 3 channels (RGB)
        unsigned char r = rgbImage[rgbIdx];
        unsigned char g = rgbImage[rgbIdx + 1];
        unsigned char b = rgbImage[rgbIdx + 2];
        grayImage[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// Kernel to compute histogram using atomic operations
__global__ void computeHistogram(const unsigned char *grayImage, int *histogram, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned char pixel = grayImage[idx];
        atomicAdd(&histogram[pixel], 1);
    }
}

// Kernel to apply histogram equalization
__global__ void applyEqualization(const unsigned char *grayImage, unsigned char *equalizedImage, unsigned char *equalizeMap, int size, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        unsigned char eqValue = equalizeMap[grayImage[idx]];
        for(int c = 0; c < channels; c++) {
            int outIdx = idx * channels + c;
            equalizedImage[outIdx] = eqValue;
        }
    }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  //Initialize CUDA device
  int deviceCount;
  wbCheck(cudaGetDeviceCount(&deviceCount));
  if (deviceCount == 0) {
      wbLog(ERROR, "No CUDA devices found.");
      return -1;
  }
  wbCheck(cudaSetDevice(0));

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);


  //@@ insert code here
  // Get pointers to the input and output image data
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  //Allocate device memory and copy input data

  int numPixels = imageWidth * imageHeight;
  int inputSize = numPixels * imageChannels * sizeof(float);
  float *deviceInput;
  unsigned char *deviceUCharInput;
  unsigned char *deviceGrayImage;
  unsigned char *deviceEqualizedImage;
  int *deviceHistogram;
  unsigned char *deviceEqualizeMap;

  // Allocate device memory for input image (float)
  wbCheck(cudaMalloc((void **)&deviceInput, inputSize));
  wbCheck(cudaMemcpy(deviceInput, hostInputImageData, inputSize, cudaMemcpyHostToDevice));

  // Allocate device memory for unsigned char input
  wbCheck(cudaMalloc((void **)&deviceUCharInput, numPixels * imageChannels * sizeof(unsigned char)));

  // Allocate device memory for grayscale image
  wbCheck(cudaMalloc((void **)&deviceGrayImage, numPixels * sizeof(unsigned char)));

  // Allocate device memory for equalized image
  wbCheck(cudaMalloc((void **)&deviceEqualizedImage, numPixels * imageChannels * sizeof(unsigned char)));

  // Allocate device memory for histogram
  wbCheck(cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(int)));
  wbCheck(cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(int)));

  // Allocate device memory for equalize map
  wbCheck(cudaMalloc((void **)&deviceEqualizeMap, HISTOGRAM_LENGTH * sizeof(unsigned char)));

  //@@ Insert code here: Define block and grid sizes
  int threadsPerBlock = 256;
  int blocksPerGridInput = (numPixels * imageChannels + threadsPerBlock - 1) / threadsPerBlock;
  int blocksPerGridGray = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
  int blocksPerGridHist = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
  int blocksPerGridEqualize = (numPixels + threadsPerBlock - 1) / threadsPerBlock;


  //@@ Insert code here: Launch CUDA kernels

  // Step 1: Cast to unsigned char
  castToUChar<<<blocksPerGridInput, threadsPerBlock>>>(deviceInput, deviceUCharInput, numPixels * imageChannels);
  wbCheck(cudaGetLastError());

  // Step 2: Convert RGB to Grayscale
  rgbToGrayscale<<<blocksPerGridGray, threadsPerBlock>>>(deviceUCharInput, deviceGrayImage, imageWidth, imageHeight);
  wbCheck(cudaGetLastError());

  // Step 3: Compute Histogram
  computeHistogram<<<blocksPerGridHist, threadsPerBlock>>>(deviceGrayImage, deviceHistogram, numPixels);
  wbCheck(cudaGetLastError());

  // Step 4: Copy histogram back to host for scan
  int hostHistogram[HISTOGRAM_LENGTH];
  wbCheck(cudaMemcpy(hostHistogram, deviceHistogram, HISTOGRAM_LENGTH * sizeof(int), cudaMemcpyDeviceToHost));

  // Compute the prefix sum (scan) on host
  int scan[HISTOGRAM_LENGTH];
  scan[0] = hostHistogram[0];
  for(int i = 1; i < HISTOGRAM_LENGTH; i++) {
      scan[i] = scan[i - 1] + hostHistogram[i];
  }

  // Find the minimum non-zero value in the scan for normalization
  int scan_min = 0;
  for(int i = 0; i < HISTOGRAM_LENGTH; i++) {
      if(scan[i] != 0) {
          scan_min = scan[i];
          break;
      }
  }

  // Create the equalization mapping on host
  unsigned char hostEqualizeMap[HISTOGRAM_LENGTH];
  float scale = 255.0f / (float)(numPixels - scan_min);
  for(int i = 0; i < HISTOGRAM_LENGTH; i++) {
      hostEqualizeMap[i] = (unsigned char)((scan[i] - scan_min) * scale);
  }

  // Copy the equalize map to device
  wbCheck(cudaMemcpy(deviceEqualizeMap, hostEqualizeMap, HISTOGRAM_LENGTH * sizeof(unsigned char), cudaMemcpyHostToDevice));

  // Step 5: Apply Histogram Equalization
  applyEqualization<<<blocksPerGridEqualize, threadsPerBlock>>>(deviceGrayImage, deviceEqualizedImage, deviceEqualizeMap, numPixels, imageChannels);
  wbCheck(cudaGetLastError());

  //@@ Insert code here: Copy the equalized image back to host and convert to float

  // Allocate temporary host memory for equalized image
  unsigned char *hostEqualizedImage = (unsigned char *)malloc(numPixels * imageChannels * sizeof(unsigned char));
  if (hostEqualizedImage == NULL) {
      wbLog(ERROR, "Failed to allocate memory for hostEqualizedImage.");
      // Free device memory before exiting
      wbCheck(cudaFree(deviceInput));
      wbCheck(cudaFree(deviceUCharInput));
      wbCheck(cudaFree(deviceGrayImage));
      wbCheck(cudaFree(deviceEqualizedImage));
      wbCheck(cudaFree(deviceHistogram));
      wbCheck(cudaFree(deviceEqualizeMap));
      return -1;
  }

  // Copy equalized image back to host
  wbCheck(cudaMemcpy(hostEqualizedImage, deviceEqualizedImage, numPixels * imageChannels * sizeof(unsigned char), cudaMemcpyDeviceToHost));

  // Convert equalized image from unsigned char [0,255] to float [0,1]
  for(int i = 0; i < numPixels * imageChannels; i++) {
      hostOutputImageData[i] = (float)hostEqualizedImage[i] / 255.0f;
  }

  //@@ Insert code here: Free device and host memory

  // Free device memory
  wbCheck(cudaFree(deviceInput));
  wbCheck(cudaFree(deviceUCharInput));
  wbCheck(cudaFree(deviceGrayImage));
  wbCheck(cudaFree(deviceEqualizedImage));
  wbCheck(cudaFree(deviceHistogram));
  wbCheck(cudaFree(deviceEqualizeMap));

  // Free host temporary memory
  free(hostEqualizedImage);

  wbSolution(args, outputImage);

  //@@ insert code here

  return 0;
}

