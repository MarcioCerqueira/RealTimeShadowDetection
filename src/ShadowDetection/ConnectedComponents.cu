#include "GPUConnectedComponents.h"

thrust::device_vector<int> thrustHashImage;
thrust::device_vector<int> thrustOutputHashImage;	
	
__global__ void discardLabels(unsigned short *labelImage, unsigned char *binaryImage, int *labelCount, size_t binaryPitch, size_t labelPitch, int rows, int cols, float omega) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;
	unsigned char *binaryImageRow = binaryImage + y * binaryPitch;
	unsigned short *labelRow = (unsigned short*)((char*)labelImage + y * labelPitch);
	if(labelCount[labelRow[x]] < (int)(rows * cols * omega)) binaryImageRow[x] = 0;
	
}

__global__ void processCCL(int *labelImage, int *hashImage, size_t labelPitch, int rows, int cols) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;
	//int *labelImageRow = (int*)((char*)labelImage + y * labelPitch);
	int *hashImageRow = (int*)((char*)hashImage + y * labelPitch);
	if(labelImage[y * cols + x] != y * cols + x) hashImageRow[x] = 0;
	else hashImageRow[x] = y * cols + x + 1;

}

__global__ void copyToThrust(int *hashImage, int *thrustImage, size_t hashPitch, int rows, int cols) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;
	int *hashImageRow = (int*)((char*)hashImage + y * hashPitch);
	thrustImage[y * cols + x] = hashImageRow[x];

}

__global__ void setZero(int *hashImage, size_t hashPitch, int rows, int cols) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;
	int *hashImageRow = (int*)((char*)hashImage + y * hashPitch);
	hashImageRow[x] = 0;

}

__global__ void loadHash(int *hashImage, int *thrustImage, size_t hashPitch, int rows, int cols, int nLabels) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows || ((y * cols + x) >= nLabels) || y * cols + x == 0) return;
	int pixel = thrustImage[y * cols + x];
	int newX = (pixel - 1) % cols;
	int newY = (pixel - 1) / cols;
	int *hashImageRow = (int*)((char*)hashImage + newY * hashPitch);
	hashImageRow[newX] = y * cols + x;
	
}

__global__ void applyCCLHash(unsigned short *shortLabelImage, int *labelImage, int *hashImage, size_t shortPitch, size_t labelPitch, int rows, int cols) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;

	//int *labelImageRow = (int*)((char*)labelImage + y * labelPitch);
	unsigned short *shortLabelImageRow = (unsigned short*)((char*)shortLabelImage + y * shortPitch);
	int label = labelImage[y * cols + x];
	if(label >= 0) {
		int newX = label % cols;
		int newY = label / cols;
		int *hashImageRow = (int*)((char*)hashImage + newY * labelPitch);
		shortLabelImageRow[x] = hashImageRow[newX];
	} else shortLabelImageRow[x] = 0;

}

struct is_not_zero
{
    __host__ __device__
    bool operator()(const int x)
    {
      return (x != 0);
    }
};

int divUp2(int a, int b) { 

    return (a + b - 1)/b;

}

void GPUCheckError2(char *methodName) {

	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) printf("%s: %s\n", methodName, cudaGetErrorString(error));
	
}

void GPUDiscardLabels(unsigned short *labelImage, unsigned char *binaryImage, int *labelCount, size_t binaryPitch, size_t labelPitch, int rows, int cols, float omega) {

	dim3 threads(16, 16);
    dim3 grid(divUp2(cols, threads.x), divUp2(rows, threads.y));
	
	discardLabels<<<grid, threads>>>(labelImage, binaryImage, labelCount, binaryPitch, labelPitch, rows, cols, omega);
	GPUCheckError2("GPUDiscardLabels");

}

void GPUProcessCCL(int *labelImage, int *hashImage, size_t labelPitch, int rows, int cols) {

	dim3 threads(16, 16);
    dim3 grid(divUp2(cols, threads.x), divUp2(rows, threads.y));
	
	processCCL<<<grid, threads>>>(labelImage, hashImage, labelPitch, rows, cols);
	GPUCheckError2("GPUProcessCCL");

}

void GPUBuildCCLHash(int *hashImage, size_t hashPitch, int nLabels, int rows, int cols) {

	dim3 threads(16, 16);
    dim3 grid(divUp2(cols, threads.x), divUp2(rows, threads.y));
	
	thrustOutputHashImage.resize(nLabels);
	copyToThrust<<<grid, threads>>>(hashImage, thrust::raw_pointer_cast(thrustHashImage.data()), hashPitch, rows, cols);
	thrust::copy_if(thrustHashImage.begin(), thrustHashImage.end(), thrustOutputHashImage.begin(), is_not_zero());
	setZero<<<grid, threads>>>(hashImage, hashPitch, rows, cols);
	loadHash<<<grid, threads>>>(hashImage, thrust::raw_pointer_cast(thrustOutputHashImage.data()), hashPitch, rows, cols, nLabels);
	GPUCheckError2("GPUBuildCCLHash");

}

void GPUApplyCCLHash(unsigned short *shortLabelImage, int *labelImage, int *hashImage, size_t shortPitch, size_t labelPitch, int rows, int cols) {

	dim3 threads(16, 16);
    dim3 grid(divUp2(cols, threads.x), divUp2(rows, threads.y));
	applyCCLHash<<<grid, threads>>>(shortLabelImage, labelImage, hashImage, shortPitch, labelPitch, rows, cols);
	GPUCheckError2("GPUApplyCCLHash");

}

void GPUSetCCLSize(int rows, int cols) {

	thrustHashImage.resize(rows * cols);
	GPUCheckError2("GPUSetCCLSize");

}