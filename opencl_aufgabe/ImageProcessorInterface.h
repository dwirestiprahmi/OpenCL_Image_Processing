#ifndef IMAGE_PROCESSOR_INTERFACE_H
#define IMAGE_PROCESSOR_INTERFACE_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

struct RGB {
    float r, g, b;
};

struct HSV {
    float h, s, v;
};

class ImageProcessorInterface {
public:
    virtual ~ImageProcessorInterface() {}

    virtual void execute(std::vector<std::string>& files, std::string& path) = 0;
    virtual void rgbToHsv(const cv::Mat& input, cv::Mat& output) = 0;
    virtual void boxBlur(const cv::Mat& input, cv::Mat& output, int kernelSize) = 0;
    virtual void runtime(std::vector<std::string>& files, std::string& path, int num_runs) = 0;
};

#endif // IMAGE_PROCESSOR_INTERFACE_H

