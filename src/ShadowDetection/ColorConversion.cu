#include "ColorConversion.h"

__constant__ float deviceCoeffs0[5];
__constant__ float deviceCoeffs[9];

__device__ int deScale(int x, int n) {
	return ((x) + (1 << ((n)-1))) >> (n);
}

__global__ void performColorConversion(unsigned char* coloredImage, unsigned char *bImage, unsigned char *CbImage, unsigned char *grayImage, ushort *LabCbrtTab_b, 
	ushort *sRGBGammaTab_b, size_t colorPitch, size_t bPitch, int rows, int cols) {

	const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= cols || y >= rows) return;

	unsigned char *coloredImageRow = coloredImage + y * colorPitch;
	unsigned char *bImageRow = bImage + y * bPitch;
	unsigned char *CbImageRow = CbImage + y * bPitch;
	unsigned char *grayImageRow = grayImage + y * bPitch;

	int red = coloredImageRow[x * 3 + 0];
	int green = coloredImageRow[x * 3 + 1];
	int blue = coloredImageRow[x * 3 + 2];
		
	int fY = LabCbrtTab_b[deScale(sRGBGammaTab_b[red] * deviceCoeffs[3] + sRGBGammaTab_b[green] * deviceCoeffs[4] + sRGBGammaTab_b[blue] * deviceCoeffs[5], lab_shift)];
    int fZ = LabCbrtTab_b[deScale(sRGBGammaTab_b[red] * deviceCoeffs[6] + sRGBGammaTab_b[green] * deviceCoeffs[7] + sRGBGammaTab_b[blue] * deviceCoeffs[8], lab_shift)];
    int b2 = deScale(200*(fY - fZ) + 128*(1 << lab_shift2), lab_shift2);
	bImageRow[x] = (unsigned char)b2; 
		 
	float Y = (blue * deviceCoeffs0[0] + green * deviceCoeffs0[1] + red * deviceCoeffs0[2]);
	grayImageRow[x] = Y;
	CbImageRow[x] = (float)(red- Y) * deviceCoeffs0[4] + 128;	

}

int divUp(int a, int b) { 

    return (a + b - 1)/b;

}

void GPUCheckError(char *methodName) {

	cudaThreadSynchronize();
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess) printf("%s: %s\n", methodName, cudaGetErrorString(error));
	
}

void GPUInitializeConstantMemory(float *coeffs, const float *coeffs0) {

	cudaMemcpyToSymbol(deviceCoeffs, coeffs, 9 * sizeof(float));
	cudaMemcpyToSymbol(deviceCoeffs0, coeffs0, 5 * sizeof(float));

}

void GPUPerformColorConversion(unsigned char* coloredImage, unsigned char *bImage, unsigned char *CbImage, unsigned char *grayImage, ushort *LabCbrtTab_b, 
	ushort *sRGBGammaTab_b, size_t colorPitch, size_t bPitch, int rows, int cols) {

	dim3 threads(16, 16);
    dim3 grid(divUp(cols, threads.x), divUp(rows, threads.y));
	
	performColorConversion<<<grid, threads>>>(coloredImage, bImage, CbImage, grayImage, LabCbrtTab_b, sRGBGammaTab_b, colorPitch, bPitch, rows, cols);
	GPUCheckError("GPUPerformColorConversion");

}