#ifndef CCOMPONENTS_H
#define CCOMPONENTS_H

#include <stdio.h>
#ifdef DETECT_SHADOW_USING_CUDA
#include "cuda_runtime.h"
#endif

//void GPUDiscardLabels(unsigned char *labelImage, unsigned char *binaryImage, int *labelCount, size_t labelPitch, size_t binaryPitch, int rows, int cols);
template<typename StatsOp> static int connectedComponents_sub1(const cv::Mat &I, cv::Mat &L, int connectivity, int ccltype, StatsOp &sop);
int connectedComponents(cv::InputArray _img, cv::OutputArray _labels, int connectivity, int ltype, int ccltype);
int connectedComponents(cv::InputArray _img, cv::OutputArray _labels, int connectivity, int ltype);
int connectedComponentsWithStats(cv::InputArray _img, cv::OutputArray _labels, cv::OutputArray statsv, cv::OutputArray centroids, int connectivity, int ltype, int ccltype);
int connectedComponentsWithStats(cv::InputArray _img, cv::OutputArray _labels, cv::OutputArray statsv, cv::OutputArray centroids, int connectivity, int ltype);

#endif