#include "ShadowDetection\ShadowDetection.h"

ShadowDetection::ShadowDetection() {

	alphaR = 1.0; alphaG = 0.9; alphaB = 1.0;
	alphaGray = 0.9; alphaCb = 0.9; alphab = 1.05;
	beta = 0.25; kappa = 16; 
	sigma = 0.49; omega = 0.006;

}

void ShadowDetection::initialize(int imageRows, int imageCols) {

#ifdef DETECT_SHADOW_USING_CUDA
	deviceGrayImage = cv::gpu::GpuMat(imageRows, imageCols, CV_8UC1);
	deviceBinaryImage = cv::gpu::GpuMat(imageRows, imageCols, CV_8UC1);
	for(int ch = 0; ch < 3; ch++) deviceRGB[ch] = cv::gpu::GpuMat(imageRows, imageCols, CV_8UC1);
	deviceCb = cv::gpu::GpuMat(imageRows, imageCols, CV_8UC1);
	deviceB = cv::gpu::GpuMat(imageRows, imageCols, CV_8UC1);
	deviceFilteredImage = cv::gpu::GpuMat(imageRows, imageCols, CV_8UC1);
	deviceContourImage = cv::gpu::GpuMat(imageRows, imageCols, CV_16U);
	deviceLabelImage = cv::gpu::createContinuous(imageRows, imageCols, CV_32S);
	deviceHashImage = cv::gpu::GpuMat(imageRows, imageCols, CV_32S);
	cv::Mat hKernel = cv::Mat::ones(kappa, 1, CV_32FC1);
	cv::Mat vKernel = cv::Mat::ones(1, kappa, CV_32FC1);
	hKernel = hKernel / kappa;
	vKernel = vKernel / kappa;
	deviceFilter = cv::gpu::createSeparableLinearFilter_GPU(CV_8UC1, CV_8UC1, hKernel, vKernel);
	GPUSetCCLSize(imageRows, imageCols);
#else
	grayImage = cv::Mat(imageRows, imageCols, CV_8UC1);
	binaryImage = cv::Mat(imageRows, imageCols, CV_8UC1);
	for(int ch = 0; ch < 3; ch++) RGB[ch] = cv::Mat(imageRows, imageCols, CV_8UC1);
	Cb = cv::Mat(imageRows, imageCols, CV_8UC1);
	b = cv::Mat(imageRows, imageCols, CV_8UC1);
	filter = cv::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(kappa, kappa));
	contourImage = cv::Mat(imageRows, imageCols, CV_16U);
	labelCountImage = cv::Mat(imageRows, imageCols, CV_32S);
#endif
	colorConversion.initialize();
	binaryImage = cv::Mat(imageRows, imageCols, CV_8UC1);

}

/* 
Pipeline to detect shadows in real time
1. Convert image from BGR to grayscale, YCrCb and Lab
2. Split R, G, B channels
3. Compute the mean values over R, G, B, gray, Cb and b channels
4. Binarize each one of those channels according to pre-defined thresholds
5. Run the shadow detection procedure
	5.1 Use B and R channels only if their mean value is high enough
	5.2 Use b channel only if it is relatively similar to the binary image (sometimes, the b channels stores the reverse of the binary image)	
6. Blur out noise through the median filter
*/	

cv::Mat ShadowDetection::run(cv::Mat originalImage) {

#ifdef DETECT_SHADOW_USING_CUDA
	deviceOriginalImage = cv::gpu::GpuMat(originalImage);
	colorConversion.deviceRun(deviceOriginalImage.ptr(), deviceB.ptr(), deviceCb.ptr(), deviceGrayImage.ptr(), deviceOriginalImage.step, deviceB.step, deviceOriginalImage.rows, deviceOriginalImage.cols);
	cv::gpu::split(deviceOriginalImage, deviceRGB);
	cv::gpu::meanStdDev(deviceGrayImage, meanValue[0], stdDev);
	cv::gpu::meanStdDev(deviceRGB[0], meanValue[1], stdDev);
	cv::gpu::meanStdDev(deviceRGB[1], meanValue[2], stdDev);
	cv::gpu::meanStdDev(deviceRGB[2], meanValue[3], stdDev);
	cv::gpu::meanStdDev(deviceCb, meanValue[4], stdDev);
	cv::gpu::meanStdDev(deviceB, meanValue[5], stdDev);
	cv::gpu::threshold(deviceGrayImage, deviceGrayImage, meanValue[0](0) * alphaGray, 255, CV_THRESH_BINARY_INV);
	cv::gpu::threshold(deviceRGB[0], deviceRGB[0], meanValue[1](0) * alphaB, 255, CV_THRESH_BINARY_INV);
	cv::gpu::threshold(deviceRGB[1], deviceRGB[1], meanValue[2](0) * alphaG, 255, CV_THRESH_BINARY_INV);
	cv::gpu::threshold(deviceRGB[2], deviceRGB[2], meanValue[3](0) * alphaR, 255, CV_THRESH_BINARY_INV);
	cv::gpu::threshold(deviceCb, deviceCb, meanValue[4](0) * alphaCb, 255, CV_THRESH_BINARY);
	cv::gpu::threshold(deviceB, deviceB, meanValue[5](0) * alphab, 255, CV_THRESH_BINARY_INV);
	cv::gpu::multiply(deviceGrayImage, deviceCb, deviceBinaryImage);
	if(meanValue[1](0) > meanValue[2](0) || meanValue[1](0) > meanValue[3](0)) cv::gpu::multiply(deviceBinaryImage, deviceRGB[0], deviceBinaryImage);
	if(meanValue[2](0) > meanValue[1](0) || meanValue[2](0) > meanValue[3](0)) cv::gpu::multiply(deviceBinaryImage, deviceRGB[1], deviceBinaryImage);
	if(meanValue[3](0) > meanValue[2](0) || meanValue[3](0) > meanValue[1](0)) cv::gpu::multiply(deviceBinaryImage, deviceRGB[2], deviceBinaryImage);
	cv::gpu::absdiff(deviceBinaryImage, deviceB, deviceCb);
	int count = cv::gpu::countNonZero(deviceCb);
	count = (originalImage.rows * originalImage.cols) - count;
	if(((float)(count) / (float)(originalImage.rows * originalImage.cols)) > beta) cv::gpu::multiply(deviceBinaryImage, deviceB, deviceBinaryImage);	
	deviceFilter->apply(deviceBinaryImage, deviceFilteredImage, cv::Rect(0, 0, binaryImage.cols, binaryImage.rows));
	cv::gpu::threshold(deviceFilteredImage, deviceBinaryImage, 255 * sigma, 255, CV_THRESH_BINARY);
	deviceBinaryImage.download(binaryImage);
	Coarse2FineCCL(deviceBinaryImage.ptr<unsigned char>(), deviceLabelImage.ptr<int>(), deviceBinaryImage.step, deviceBinaryImage.rows, deviceBinaryImage.cols);
	GPUProcessCCL(deviceLabelImage.ptr<int>(), deviceHashImage.ptr<int>(), deviceLabelImage.step, deviceLabelImage.rows, deviceLabelImage.cols);
	int nLabels = cv::gpu::countNonZero(deviceHashImage);
	GPUBuildCCLHash(deviceHashImage.ptr<int>(), deviceHashImage.step, nLabels, deviceHashImage.rows, deviceHashImage.cols);
	GPUApplyCCLHash(deviceContourImage.ptr<unsigned short>(), deviceLabelImage.ptr<int>(), deviceHashImage.ptr<int>(), deviceContourImage.step, deviceHashImage.step, deviceHashImage.rows, deviceHashImage.cols);
	cv::gpu::histEven(deviceContourImage, deviceHistogram, buffer, nLabels, 0, nLabels);
	GPUDiscardLabels(deviceContourImage.ptr<unsigned short>(), deviceBinaryImage.data, deviceHistogram.ptr<int>(), deviceBinaryImage.step, deviceContourImage.step, deviceBinaryImage.rows, deviceBinaryImage.cols, omega);
	deviceBinaryImage.download(binaryImage);
#else
	colorConversion.run(originalImage.ptr<unsigned char>(), b.ptr<unsigned char>(), Cb.ptr<unsigned char>(), grayImage.ptr<unsigned char>(), originalImage.rows, originalImage.cols);		
	cv::split(originalImage, RGB);
	meanValue[0] = cv::mean(grayImage);
	meanValue[1] = cv::mean(RGB[0]);
	meanValue[2] = cv::mean(RGB[1]);
	meanValue[3] = cv::mean(RGB[2]);
	meanValue[4] = cv::mean(Cb);
	meanValue[5] = cv::mean(b);
	cv::threshold(grayImage, grayImage, meanValue[0](0) * alphaGray, 255, CV_THRESH_BINARY_INV);
	cv::threshold(RGB[0], RGB[0], meanValue[1](0) * alphaB, 255, CV_THRESH_BINARY_INV);
	cv::threshold(RGB[1], RGB[1], meanValue[2](0) * alphaG, 255, CV_THRESH_BINARY_INV);
	cv::threshold(RGB[2], RGB[2], meanValue[3](0) * alphaR, 255, CV_THRESH_BINARY_INV);
	cv::threshold(Cb, Cb, meanValue[4](0) * alphaCb, 255, CV_THRESH_BINARY);
	cv::threshold(b, b, meanValue[5](0) * alphab, 255, CV_THRESH_BINARY_INV);
	binaryImage = grayImage.mul(Cb);
	if(meanValue[1](0) > meanValue[2](0) || meanValue[1](0) > meanValue[3](0)) binaryImage = binaryImage.mul(RGB[0]);
	if(meanValue[2](0) > meanValue[1](0) || meanValue[2](0) > meanValue[3](0)) binaryImage = binaryImage.mul(RGB[1]);
	if(meanValue[3](0) > meanValue[2](0) || meanValue[3](0) > meanValue[1](0)) binaryImage = binaryImage.mul(RGB[2]);
	cv::compare(binaryImage, b, contourImage, CV_CMP_EQ);
	int count = cv::countNonZero(contourImage);
	if(((float)count / (float)(originalImage.rows * originalImage.cols)) > beta) binaryImage = binaryImage.mul(b);
	filter->apply(binaryImage, binaryImage, cv::Rect(0, 0, binaryImage.cols, binaryImage.rows));
	cv::threshold(binaryImage, binaryImage, 255 * sigma, 255, CV_THRESH_BINARY);
	//Use contour detection to further reduce noise
	int nLabels = connectedComponents(binaryImage, contourImage, 8, CV_16U, 1);
	for(int l = 0; l < nLabels; l++) labelCountImage.ptr<int>()[l] = 0;
	for(int y=0; y < binaryImage.rows; y++) for(int x=0; x < binaryImage.cols; x++) labelCountImage.ptr<int>()[(int)contourImage.at<unsigned short>(y, x)]++;
	for(int y=0; y < binaryImage.rows; y++)	for(int x=0; x < binaryImage.cols; x++) if(labelCountImage.ptr<int>()[(int)contourImage.at<unsigned short>(y, x)] < binaryImage.rows * binaryImage.cols * omega) binaryImage.at<unsigned char>(y, x) = 0;	
#endif
	return binaryImage;

}