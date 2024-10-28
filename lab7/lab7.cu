// Histogram Equalization

#include <wb.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HISTOGRAM_LENGTH 256

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
        // Clamp the values to [0,1] before scaling
        float val = input[idx];
        val = val < 0.0f ? 0.0f : (val > 1.0f ? 1.0f : val);
        output[idx] = (unsigned char)(val * 255.0f);
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
        // Using the luminosity method coefficients
        grayImage[idx] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
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
__global__ void applyEqualization(const unsigned char *inputImage, unsigned char *outputImage, const unsigned char *equalizeMap, int size, int channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size * channels) {
        unsigned char pixel = inputImage[idx];
        unsigned char eqPixel = equalizeMap[pixel];
        outputImage[idx] = eqPixel;
    }
}

// Kernel to cast unsigned char image data [0,255] to float [0,1]
__global__ void castToFloat(const unsigned char *input, float *output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = ((float)input[idx]) / 255.0f;
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

    //@@ Insert code here: Initialize CUDA device
    // Initialize CUDA device and check for errors
    int deviceCount;
    wbCheck(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        wbLog(ERROR, "No CUDA devices found.");
        return -1;
    }
    wbCheck(cudaSetDevice(0)); // Select the first CUDA device

    // Read input arguments
    args = wbArg_read(argc, argv); /* parse the input arguments */

    // Get the input image file path
    inputImageFile = wbArg_getInputFile(args, 0);

    // Import data and retrieve image dimensions and channels
    inputImage = wbImport(inputImageFile);
    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    // Create an output image with the same dimensions and channels
    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    // Get pointers to the input and output image data
    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    //@@ Insert code here: Allocate device memory and copy input data

    // Calculate the number of pixels and sizes for memory allocations
    int numPixels = imageWidth * imageHeight;
    int inputSize = numPixels * imageChannels * sizeof(float);
    float *deviceInput;
    unsigned char *deviceUCharInput;
    unsigned char *deviceGrayImage;
    unsigned char *deviceEqualizedImage;
    int *deviceHistogram;
    unsigned char *deviceEqualizeMap;
    float *deviceOutputFloat;

    // Allocate device memory for input image (float)
    wbCheck(cudaMalloc((void **)&deviceInput, inputSize));
    // Copy input image data from host to device
    wbCheck(cudaMemcpy(deviceInput, hostInputImageData, inputSize, cudaMemcpyHostToDevice));

    // Allocate device memory for unsigned char input
    wbCheck(cudaMalloc((void **)&deviceUCharInput, numPixels * imageChannels * sizeof(unsigned char)));

    // Allocate device memory for grayscale image
    wbCheck(cudaMalloc((void **)&deviceGrayImage, numPixels * sizeof(unsigned char)));

    // Allocate device memory for equalized image
    wbCheck(cudaMalloc((void **)&deviceEqualizedImage, numPixels * imageChannels * sizeof(unsigned char)));

    // Allocate device memory for histogram and initialize to zero
    wbCheck(cudaMalloc((void **)&deviceHistogram, HISTOGRAM_LENGTH * sizeof(int)));
    wbCheck(cudaMemset(deviceHistogram, 0, HISTOGRAM_LENGTH * sizeof(int)));

    // Allocate device memory for equalize map
    wbCheck(cudaMalloc((void **)&deviceEqualizeMap, HISTOGRAM_LENGTH * sizeof(unsigned char)));

    // Allocate device memory for output float image
    wbCheck(cudaMalloc((void **)&deviceOutputFloat, inputSize));
    //@@ Insert code here: Define block and grid sizes

    // Define the number of threads per block
    int threadsPerBlock = 256;

    // Calculate the number of blocks per grid for each kernel
    int blocksPerGridInput = (numPixels * imageChannels + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridGray = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridHist = (numPixels + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridEqualize = (numPixels * imageChannels + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridCast = (numPixels * imageChannels + threadsPerBlock - 1) / threadsPerBlock;
    
    //@@ Insert code here: Launch CUDA kernels

    // Step 1: Cast float image to unsigned char
    castToUChar<<<blocksPerGridInput, threadsPerBlock>>>(deviceInput, deviceUCharInput, numPixels * imageChannels);
    wbCheck(cudaGetLastError());

    // Step 2: Convert RGB image to Grayscale
    rgbToGrayscale<<<blocksPerGridGray, threadsPerBlock>>>(deviceUCharInput, deviceGrayImage, imageWidth, imageHeight);
    wbCheck(cudaGetLastError());

    // Step 3: Compute Histogram of the Grayscale image
    computeHistogram<<<blocksPerGridHist, threadsPerBlock>>>(deviceGrayImage, deviceHistogram, numPixels);
    wbCheck(cudaGetLastError());

    // Step 4: Copy histogram back to host for scan (CDF computation)
    int hostHistogram[HISTOGRAM_LENGTH];
    wbCheck(cudaMemcpy(hostHistogram, deviceHistogram, HISTOGRAM_LENGTH * sizeof(int), cudaMemcpyDeviceToHost));

    // Compute the prefix sum (scan) on host to obtain CDF
    int scan[HISTOGRAM_LENGTH];
    scan[0] = hostHistogram[0];
    for(int i = 1; i < HISTOGRAM_LENGTH; i++) {
        scan[i] = scan[i - 1] + hostHistogram[i];
    }

    // Step 5: Find the minimum non-zero value in the CDF for normalization
    int scan_min = 0;
    for(int i = 0; i < HISTOGRAM_LENGTH; i++) {
        if(scan[i] != 0) {
            scan_min = scan[i];
            break;
        }
    }

    // Step 6: Create the histogram equalization mapping function
    unsigned char hostEqualizeMap[HISTOGRAM_LENGTH];
    float scale = 255.0f / (float)(numPixels - scan_min);
    for(int i = 0; i < HISTOGRAM_LENGTH; i++) {
        // Applying the histogram equalization formula
        float cdf = (float)(scan[i] - scan_min) / (float)(numPixels - scan_min);
        // Clamp the values to [0, 1]
        if(cdf < 0.0f) cdf = 0.0f;
        if(cdf > 1.0f) cdf = 1.0f;
        hostEqualizeMap[i] = (unsigned char)(cdf * 255.0f);
    }

    // Copy the equalize map to device memory
    wbCheck(cudaMemcpy(deviceEqualizeMap, hostEqualizeMap, HISTOGRAM_LENGTH * sizeof(unsigned char), cudaMemcpyHostToDevice));

    // Step 7: Apply Histogram Equalization to the image
    applyEqualization<<<blocksPerGridEqualize, threadsPerBlock>>>(deviceUCharInput, deviceEqualizedImage, deviceEqualizeMap, numPixels, imageChannels);
    wbCheck(cudaGetLastError());

    // Step 8: Cast the equalized image back to float [0,1]
    castToFloat<<<blocksPerGridCast, threadsPerBlock>>>(deviceEqualizedImage, deviceOutputFloat, numPixels * imageChannels);
    wbCheck(cudaGetLastError());

    //@@ Insert code here: Copy the equalized image back to host and convert to float


    // Copy the final float image back to host memory
    wbCheck(cudaMemcpy(hostOutputImageData, deviceOutputFloat, numPixels * imageChannels * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    wbCheck(cudaFree(deviceInput));
    wbCheck(cudaFree(deviceUCharInput));
    wbCheck(cudaFree(deviceGrayImage));
    wbCheck(cudaFree(deviceEqualizedImage));
    wbCheck(cudaFree(deviceHistogram));
    wbCheck(cudaFree(deviceEqualizeMap));
    wbCheck(cudaFree(deviceOutputFloat));

    // Write the solution to the output
    wbSolution(args, outputImage);

    //@@ Insert code here

    return 0;
}
