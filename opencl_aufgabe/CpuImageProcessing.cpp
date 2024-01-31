#include "CpuImageProcessing.h"
#include "OpenCVImageProcessing.h"

CpuImageProcessing::CpuImageProcessing() {}

CpuImageProcessing::~CpuImageProcessing() {}

CpuImageProcessing::HSV CpuImageProcessing::rgbToHsvCPU(float r, float g, float b) {
    // Normalized to the range [0, 1]
    float red = r / 255.0f;
    float green = g / 255.0f;
    float blue = b / 255.0f;

    // Finding max and min values
    float maxVal = std::max(red, std::max(green, blue));
    float minVal = std::min(red, std::min(green, blue));
    
    float delta = maxVal - minVal;
    float h = 0, s = 0, v = maxVal;

    // Calculate hue
    if (delta == 0) {
        h = 0;
    }
    else if (maxVal == red) {
        h = 60 * ((green - blue) / delta);
    }
    else if (maxVal == green) {
        h = 60 * ((blue - red) / delta) + 120;
    }
    else {
        h = 60 * ((red - green) / delta) + 240;
    }

    if (h < 0)
        h += 360;

    // Calculate saturation
    s = (maxVal == 0) ? 0 : (delta / maxVal);

    return { h, s, v };
}

void CpuImageProcessing::rgbToHsv(const cv::Mat& input, cv::Mat& output) {
    std::vector<std::vector<RGB>> rgbImage(input.rows, std::vector<RGB>(input.cols));
    std::vector<std::vector<HSV>> hsvImage;

    // Convert OpenCV Mat to RGB vector of vectors
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            cv::Vec3b pixel = input.at<cv::Vec3b>(i, j);
            rgbImage[i][j] = 
            { static_cast<float>(pixel[0]), 
                static_cast<float>(pixel[1]), 
                static_cast<float>(pixel[2]) 
            };
        }
    }

    // Make sure that RGB image is not empty
    assert(!rgbImage.empty());

    // Set the size of HSV image to have the same size as the RGB image
    hsvImage.resize(rgbImage.size(), std::vector<HSV>(rgbImage[0].size()));
    
    // Conversion to HSV on each pixel
    for (size_t i = 0; i < rgbImage.size(); ++i) {
        for (size_t j = 0; j < rgbImage[i].size(); ++j) {
            hsvImage[i][j] = rgbToHsvCPU(
                rgbImage[i][j].r, 
                rgbImage[i][j].g, 
                rgbImage[i][j].b
            );
        }
    }

    // Mapping HSV to Output Matrix
    for (size_t i = 0; i < hsvImage.size(); ++i) {
        for (size_t j = 0; j < hsvImage[i].size(); ++j) {
            HSV hsvPixel = hsvImage[i][j];
            cv::Vec3b& hsvMatPixel = output.at<cv::Vec3b>(i, j);
            
            // Converting HSV to OpenCV Format
            hsvMatPixel = { 
                static_cast<uchar>(hsvPixel.h / 2.0f), 
                static_cast<uchar>(hsvPixel.s * 255.0f), 
                static_cast<uchar>(hsvPixel.v * 255.0f) 
            };
        }
    }
}

void CpuImageProcessing::boxBlur(const cv::Mat& inputImage, cv::Mat& outputImage, int kernelSize) {
    const int depth = inputImage.channels();
    int divider = ((2 * kernelSize + 1) * (2 * kernelSize + 1));

    for (int channels = 0; channels < depth; ++channels) { // Iterate over channels
        for (int posx = 0; posx < inputImage.cols; ++posx) {
            for (int posy = 0; posy < inputImage.rows; ++posy) {
                float sum = 0.0f;

                for (int i = -kernelSize; i <= kernelSize; ++i) {
                    for (int j = -kernelSize; j <= kernelSize; ++j) {
                        int x = std::max(0, std::min(posx + i, inputImage.cols - 1));
                        int y = std::max(0, std::min(posy + j, inputImage.rows - 1));

                        if (x >= 0 && x < inputImage.cols && y >= 0 && y < inputImage.rows) {
                            sum += inputImage.at<cv::Vec3b>(y, x)[channels];
                        }
                    }
                }

                float result = sum / divider;
                outputImage.at<cv::Vec3b>(posy, posx)[channels] = static_cast<uchar>(result);
            }
        }
    }
}

void CpuImageProcessing::runtime(std::vector<std::string>& files, std::string& path, int num_runs) {
    //Vector to store durations
    std::vector<std::chrono::duration<double>> durationsHSV;
    std::vector<std::chrono::duration<double>> durationsBlur;

    for (int i = 0; i < files.size(); i++) {
        for (int n = 0; n < num_runs; ++n) {
            cv::Mat inputImage = cv::imread(path + files.at(i));
            cv::Mat hsvImage = cv::Mat::zeros(inputImage.size(), inputImage.type());
            cv::Mat blurredImage = cv::Mat::zeros(inputImage.size(), inputImage.type());

            // Record the starting time
            auto start = std::chrono::high_resolution_clock::now();

            // Convert RGB image to HSV image
            rgbToHsv(inputImage, hsvImage);
           
            // Record the ending time
            auto end = std::chrono::high_resolution_clock::now();

            // Calculate the duration and add it to the vector
            durationsHSV.push_back(end - start);

            start = std::chrono::high_resolution_clock::now();

            boxBlur(inputImage, blurredImage, 10);

            end = std::chrono::high_resolution_clock::now();

            // Calculate the duration and add it to the vector
            durationsBlur.push_back(end - start);
        }

        // Calculate the total duration
        std::chrono::duration<double> total_duration_hsv = std::chrono::duration<double>::zero();
        for (const auto& duration : durationsHSV) {
            total_duration_hsv += duration;
        }

        // Calculate the total duration
        std::chrono::duration<double> total_duration_blur = std::chrono::duration<double>::zero();
        for (const auto& duration : durationsBlur) {
            total_duration_blur += duration;
        }

        // Calculate the average duration
        double average_duration = total_duration_hsv.count() / num_runs;

        // Prepare to write runtime into .txt file
        std::ofstream myfile;
        myfile.open("runtimeEvaluation.txt", std::fstream::app);

        std::string notifyHsvRuntime = "Average Runtime HSV With CPU, Picture " + std::to_string(i + 1) + ": " + std::to_string(average_duration) + " seconds\n";

        // Output the average duration
        std::cout << notifyHsvRuntime;
        myfile << notifyHsvRuntime;

        average_duration = total_duration_blur.count() / num_runs;

        std::string notifyBlurRuntime = "Average Runtime Blur With CPU, Picture " + std::to_string(i + 1) + ": " + std::to_string(average_duration) + " seconds\n";
        // Output the average duration
        std::cout << notifyBlurRuntime;
        myfile << notifyBlurRuntime;

        durationsHSV.clear();
        durationsBlur.clear();
        myfile.close();
    }
}

void CpuImageProcessing::execute(std::vector<std::string>& files, std::string& path) {
    // To access the image processing operation with OpenCV
    OpenCVImageProcessing ocvip;

    // Process all of the images that are included in the files parameter
    for (int i = 0; i < files.size(); i++) {
        // Variables for original, hsv, and blurred image
        cv::Mat inputImage = cv::imread(path + files.at(i));
        cv::Mat hsvImage = cv::Mat::zeros(inputImage.size(), inputImage.type());
        cv::Mat blurImage = cv::Mat::zeros(inputImage.size(), inputImage.type());
        cv::Mat blurHSVImage = cv::Mat::zeros(inputImage.size(), inputImage.type());
        
        // Convert RGB image to HSV image
        rgbToHsv(inputImage, hsvImage);

        // Blur Original Image
        int kernelSize = 10;
        boxBlur(inputImage, blurImage, kernelSize);

        // Blur HSV Image
        boxBlur(hsvImage, blurHSVImage, kernelSize);
        std::cout << "Finished processing image " << i + 1 << " with CPU." << std::endl;
        
        // Display the results
        cv::imshow("Original Image", inputImage);
        cv::imshow("HSV Image", hsvImage);
        cv::imshow("Blurred Original Image", blurImage);
        cv::imshow("Blurred HSV Image ", blurHSVImage);
        cv::waitKey(0);
        cv::destroyAllWindows();

        // Compare the custom result with the already existing operation from OpenCV
        // Variables for hsv and blur image with OpenCV
        cv::Mat hsvImageOpenCV = cv::Mat::zeros(inputImage.size(), inputImage.type());
        cv::Mat blurImageOpenCV = cv::Mat::zeros(inputImage.size(), inputImage.type());
        cv::Mat blurHSVImageOpenCV = cv::Mat::zeros(inputImage.size(), inputImage.type());
        
        // HSV and Blur with OpenCV
        ocvip.rgbToHsv(inputImage, hsvImageOpenCV);
        ocvip.boxBlur(inputImage, blurImageOpenCV, kernelSize);
        ocvip.boxBlur(hsvImageOpenCV, blurHSVImageOpenCV, kernelSize);

        // Compare the results
        cv::Mat diff_hsv_image = hsvImage - hsvImageOpenCV;
        cv::Mat diff_blur_image = blurImage - blurImageOpenCV;
        cv::Mat diff_hsv_blur_image = blurHSVImage - blurHSVImageOpenCV;
        cv::imshow("Diff HSV Image", diff_hsv_image);
        cv::imshow("Diff Blur Image", diff_blur_image);
        cv::imshow("Diff HSV Blur Image", diff_hsv_blur_image);
        cv::waitKey(0);
        cv::destroyAllWindows();

        // Specify the folder path to save the images
        std::string folderPath = "Results CPU\\";

        // Create .jpg file from the result
        std::string numberingFile = std::to_string(i + 1);
        std::string hsvImageFile = folderPath + numberingFile + ".hsvImage.jpg";
        std::string blurredImageFile = folderPath + numberingFile + ".blurredImage.jpg";
        std::string blurredHSVImageFile = folderPath + numberingFile + ".blurredHSVImage.jpg";
        cv::imwrite(hsvImageFile, hsvImage);
        cv::imwrite(blurredImageFile, blurImage);
        cv::imwrite(blurredHSVImageFile, blurHSVImage);
    }
}