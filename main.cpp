#include "ShadowDetection\ShadowDetection.h"

int main(int argc, char **argv) {
	
	if(argc != 2) {
		printf("Usage: ShadowDetection.exe imagefile.extension\n");
		return 0;
	}

	cv::Mat inputImage = cv::imread(argv[1]);
	cv::Mat outputImage;

	ShadowDetection shadowDetection;
	shadowDetection.initialize(inputImage.rows, inputImage.cols);
	outputImage = shadowDetection.run(inputImage);
	
	while(cv::waitKey(33) != 13) {
		cv::imshow("Input Image", inputImage);
		cv::imshow("Output Image", outputImage);
	}
	
	return 0;

}
