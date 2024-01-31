#ifndef OPENCV_IMAGE_PROCESSING_H
#define OPENCV_IMAGE_PROCESSING_H

#include "ImageProcessorInterface.h"

class OpenCVImageProcessing : public ImageProcessorInterface{
public:
	OpenCVImageProcessing();
	virtual ~OpenCVImageProcessing();

	virtual void execute(std::vector<std::string>& files, std::string& path) override;
	virtual void rgbToHsv(const cv::Mat& input, cv::Mat& output) override;
	virtual void boxBlur(const cv::Mat& input, cv::Mat& output, int kernelSize) override;
	virtual void runtime(std::vector<std::string>& files, std::string& path, int num_runs) override;

};

#endif // OPENCV_IMAGE_PROCESSING_H
