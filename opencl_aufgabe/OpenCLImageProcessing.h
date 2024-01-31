#ifndef OPENCL_IMAGE_PROCESSING_H
#define OPENCL_IMAGE_PROCESSING_H

#include <CL/cl.hpp>
#include "ImageProcessorInterface.h"

class OpenCLImageProcessing : public ImageProcessorInterface {
public:
	OpenCLImageProcessing();
	virtual ~OpenCLImageProcessing();

	virtual void execute(std::vector<std::string>& files, std::string& path) override;
	virtual void rgbToHsv(const cv::Mat& input, cv::Mat& output) override;
	virtual void boxBlur(const cv::Mat& input, cv::Mat& output, int kernelSize) override;
	virtual void runtime(std::vector<std::string>& files, std::string& path, int num_runs) override;

private:
	cl::Context context;
	cl::CommandQueue commandQueue;
	cl::Program program;
	cl::Device device;

	std::string read_kernel(const char* filename);
};

#endif // OPENCL_IMAGE_PROCESSING_H