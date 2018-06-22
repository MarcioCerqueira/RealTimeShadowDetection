#ifndef COLOR_CONVERSION_H
#define COLOR_CONVERSION_H

#include <opencv2\opencv.hpp>
#ifdef DETECT_SHADOW_USING_CUDA
#include <cuda_runtime.h>
#endif

#undef lab_shift
#define lab_shift 12
#define gamma_shift 3
#define lab_shift2 (lab_shift + gamma_shift)
#define LAB_CBRT_TAB_SIZE_B (256*3/2*(1<<gamma_shift))
#define CV_DESCALE(x,n) (((x) + (1 << ((n)-1))) >> (n))

static const float sRGB2XYZ_D65[] =
{
    0.412453f, 0.357580f, 0.180423f,
    0.212671f, 0.715160f, 0.072169f,
    0.019334f, 0.119193f, 0.950227f
};

static const float D65[] = { 0.950456f, 1.f, 1.088754f };
static const float coeffs0[] = {0.299f, 0.587f, 0.114f, 0.713f, 0.564f};
static ushort sRGBGammaTab_b[256]; 
static ushort LabCbrtTab_b[LAB_CBRT_TAB_SIZE_B];

#ifdef DETECT_SHADOW_USING_CUDA
void GPUInitializeConstantMemory(float *coeffs, const float *coeffs0);
void GPUPerformColorConversion(unsigned char* coloredImage, unsigned char *bImage, unsigned char *CbImage, unsigned char *grayImage, ushort *LabCbrtTab_b, 
	ushort *sRGBGammaTab_b, size_t colorPitch, size_t bPitch, int rows, int cols);
#endif

class ColorConversion
{
public:
	ColorConversion();
	~ColorConversion();
	void initialize();
	void run(unsigned char* coloredImage, unsigned char *bImage, unsigned char *CbImage, unsigned char *grayImage, int rows, int cols);
#ifdef DETECT_SHADOW_USING_CUDA
	void deviceRun(unsigned char* coloredImage, unsigned char *bImage, unsigned char *CbImage, unsigned char *grayImage, size_t colorPitch, size_t bPitch, int rows, 
		int cols);
#endif
private:
	
	float scale[3];
	float coeffs[9];
	
	ushort *deviceLabCbrtTab_b;
	ushort *devicesRGBGammaTab_b;
	float *deviceCoeffs;
};
#endif