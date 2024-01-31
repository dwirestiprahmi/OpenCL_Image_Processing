#ifndef CPU_IMAGE_PROCESSING_H
#define CPU_IMAGE_PROCESSING_H

#include "ImageProcessorInterface.h"

class CpuImageProcessing : public ImageProcessorInterface {
public:
    CpuImageProcessing();
    virtual ~CpuImageProcessing();

    struct RGB {
        float r, g, b;
    };

    struct HSV {
        float h, s, v;
    };

    virtual void execute(std::vector<std::string>& files, std::string& path) override;
    virtual void rgbToHsv(const cv::Mat& input, cv::Mat& output) override;
    virtual void boxBlur(const cv::Mat& input, cv::Mat& output, int kernelSize) override;
    virtual void runtime(std::vector<std::string>& files, std::string& path, int num_runs) override;

private:
    HSV rgbToHsvCPU(float r, float g, float b);
};

#endif // CPU_IMAGE_PROCESSING_H

