#include "ShadowDetection\ColorConversion.h"

ColorConversion::ColorConversion() {
}

ColorConversion::~ColorConversion() {
	
#ifdef DETECT_SHADOW_USING_CUDA
	cudaFree(deviceLabCbrtTab_b);
	cudaFree(devicesRGBGammaTab_b);
	cudaFree(deviceCoeffs);
#endif

}

void ColorConversion::initialize() {
	
	for(int i = 0; i < LAB_CBRT_TAB_SIZE_B; i++)
    {
		float x = i*(1.f/(255.f*(1 << 3)));
	    LabCbrtTab_b[i] = (ushort)((1 << lab_shift2)*(x < 0.008856f ? x*7.787f + 0.13793103448275862f : cvCbrt(x)));
	}

	scale[0] = (1 << lab_shift)/D65[0];
	scale[1] = (float)(1 << lab_shift);
	scale[2] = (1 << lab_shift)/D65[2];

	for( int i = 0; i < 3; i++ )
    {
		coeffs[i*3+2] = cvRound(sRGB2XYZ_D65[i*3]*scale[i]);
        coeffs[i*3+1] = cvRound(sRGB2XYZ_D65[i*3+1]*scale[i]);
        coeffs[i*3+0] = cvRound(sRGB2XYZ_D65[i*3+2]*scale[i]);
    }

	for(int i = 0; i < 256; i++)
    {
        float x = i*(1.f/255.f);
        sRGBGammaTab_b[i] = (ushort)(255.f*(1 << gamma_shift)*(x <= 0.04045f ? x*(1.f/12.92f) : (float)pow((double)(x + 0.055)*(1./1.055), 2.4)));
    }

#ifdef DETECT_SHADOW_USING_CUDA
	cudaMalloc(&deviceLabCbrtTab_b, LAB_CBRT_TAB_SIZE_B * sizeof(ushort));
	cudaMalloc(&devicesRGBGammaTab_b, 256 * sizeof(ushort));
	cudaMalloc(&deviceCoeffs, 9 * sizeof(float));
	//cudaMalloc(&deviceCoeffs0, 5 * sizeof(float));

	cudaMemcpy(deviceLabCbrtTab_b, LabCbrtTab_b, LAB_CBRT_TAB_SIZE_B * sizeof(ushort), cudaMemcpyHostToDevice);
	cudaMemcpy(devicesRGBGammaTab_b, sRGBGammaTab_b, 256 * sizeof(ushort), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceCoeffs, coeffs, 9 * sizeof(float), cudaMemcpyHostToDevice);
	GPUInitializeConstantMemory(coeffs, coeffs0);
#endif

}

void ColorConversion::run(unsigned char* coloredImage, unsigned char *bImage, unsigned char *CbImage, unsigned char *grayImage, int rows, int cols) {

	for(int pixel = 0; pixel < rows * cols; pixel++) {
		int red = coloredImage[pixel * 3 + 0];
		int green = coloredImage[pixel * 3 + 1];
		int blue = coloredImage[pixel * 3 + 2];
		int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
        C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
        C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];
		int fY = LabCbrtTab_b[(int)CV_DESCALE(sRGBGammaTab_b[red]*C3 + sRGBGammaTab_b[green]*C4 + sRGBGammaTab_b[blue]*C5, lab_shift)];
        int fZ = LabCbrtTab_b[(int)CV_DESCALE(sRGBGammaTab_b[red]*C6 + sRGBGammaTab_b[green]*C7 + sRGBGammaTab_b[blue]*C8, lab_shift)];
        int b2 = CV_DESCALE( 200*(fY - fZ) + 128*(1 << lab_shift2), lab_shift2 );
		bImage[pixel] = (unsigned char)b2; 
		float Y = (blue * coeffs0[0] + green * coeffs0[1] + red * coeffs0[2]);
		grayImage[pixel] = Y;
		CbImage[pixel] = (float)(red- Y) * coeffs0[4] + 128;	
	}

}

#ifdef DETECT_SHADOW_USING_CUDA
void ColorConversion::deviceRun(unsigned char* coloredImage, unsigned char *bImage, unsigned char *CbImage, unsigned char *grayImage, size_t colorPitch, size_t bPitch, 
	int rows, int cols) {

	GPUPerformColorConversion(coloredImage, bImage, CbImage, grayImage, deviceLabCbrtTab_b, devicesRGBGammaTab_b, colorPitch, bPitch, rows, cols);

}
#endif