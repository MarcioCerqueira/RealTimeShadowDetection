#ifndef GPUCONNECTEDCOMPONENTS_H
#define GPUCONNECTEDCOMPONENTS_H

#include <stdio.h>
#include <vector>
#include <opencv2\opencv.hpp>
#include <thrust\device_ptr.h>
#include <thrust\copy.h>
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include "cuda_runtime.h"

void Coarse2FineCCL(unsigned char *srcImg, int *labelMap, size_t binaryPitch, int rows, int cols);
void GPUDiscardLabels(unsigned short *labelImage, unsigned char *binaryImage, int *labelCount, size_t binaryPitch, size_t labelPitch, int rows, int cols, float omega);
void GPUProcessCCL(int *labelImage, int *hashImage, size_t labelPitch, int rows, int cols);
void GPUBuildCCLHash(int *hashImage, size_t hashPitch, int nLabels, int rows, int cols); 
void GPUApplyCCLHash(unsigned short *shortLabelImage, int *labelImage, int *hashImage, size_t shortPitch, size_t labelPitch, int rows, int cols);	
void GPUSetCCLSize(int rows, int cols);
std::vector<int> cuda_ccl(std::vector<int>& image, int W, int degree_of_connectivity, int threshold);

#endif