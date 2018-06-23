#ifndef SHADOW_DETECTION_H
#define SHADOW_DETECTION_H

#include <opencv2\opencv.hpp>
#include "ColorConversion.h"
#include "ConnectedComponents.h"
#ifdef DETECT_SHADOW_USING_CUDA
#include <nppi.h>
#include <opencv2\gpu\gpu.hpp>
#include "GPUConnectedComponents.h"
#endif

class ShadowDetection
{
public:
	ShadowDetection();
	void initialize(int imageRows, int imageCols);
	cv::Mat run(cv::Mat image);
	void setAlphaR(float alphaR) {this->alphaR = alphaR; }
	void setAlphaG(float alphaG) {this->alphaG = alphaG; }
	void setAlphaB(float alphaB) {this->alphaB = alphaB; }
	void setAlphaGray(float alphaGray) {this->alphaGray = alphaGray; }
	void setAlphaCb(float alphaCb) {this->alphaCb = alphaCb; }
	void setAlphab(float alphab) {this->alphab = alphab; }
	void setBeta(float beta) {this->beta = beta; }
	void setKappa(float kappa) {this->kappa = kappa; }
	void setSigma(float sigma) {this->sigma = sigma; }
	void setOmega(float omega) {this->omega = omega; }
private:

#ifdef DETECT_SHADOW_USING_CUDA
	cv::gpu::GpuMat deviceOriginalImage, deviceGroundTruthImage, deviceGrayImage, deviceBinaryImage, deviceContourImage, deviceCb, deviceB, deviceRGB[3];
	cv::gpu::GpuMat deviceFilteredImage, deviceHistogram, buffer, deviceLabelImage, deviceHashImage;
	cv::Ptr<cv::gpu::FilterEngine_GPU> deviceFilter;
#else
	cv::Mat originalImage, grayImage, contourImage, Cb, b, RGB[3], labelCountImage;
	cv::Ptr<cv::FilterEngine> filter;
#endif
	ColorConversion colorConversion;
	cv::Scalar meanValue[10], stdDev;
	cv::Mat binaryImage;
	float alphaR;
	float alphaG;
	float alphaB;
	float alphaGray;
	float alphaCb;
	float alphab;
	float beta;
	float kappa;
	float sigma;
	float omega;
};
#endif