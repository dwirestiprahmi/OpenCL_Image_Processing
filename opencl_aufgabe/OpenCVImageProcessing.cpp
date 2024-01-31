#include "OpenCVImageProcessing.h"

OpenCVImageProcessing::OpenCVImageProcessing() {}

OpenCVImageProcessing::~OpenCVImageProcessing() {}

void OpenCVImageProcessing::rgbToHsv(const cv::Mat& input, cv::Mat& output) {
    // Convert RGB to HSV
    cvtColor(input, output, cv::COLOR_RGB2HSV);
}

void OpenCVImageProcessing::boxBlur(const cv::Mat& input, cv::Mat& output, int kernelSize) {
    // Create a kernel for box filter
    cv::Mat kernel = cv::Mat::ones(kernelSize * 2 + 1, kernelSize * 2 + 1, CV_32F) / ((kernelSize * 2 + 1) * (kernelSize * 2 + 1));

    // Apply box filter
    cv::filter2D(input, output, -1, kernel);
}

void OpenCVImageProcessing::runtime(std::vector<std::string>& files, std::string& path, int num_runs){
    //Vector to store durations
    std::vector<std::chrono::duration<double>> durationsHSV;
    std::vector<std::chrono::duration<double>> durationsBlur;

    for (int i = 0; i < files.size(); i++) {

        for (int j = 0; j < num_runs; ++j) {
            cv::Mat inputImage = cv::imread(path + files.at(i), cv::IMREAD_UNCHANGED);
            cv::Mat hsvImage = cv::Mat::zeros(inputImage.size(), inputImage.type());
            cv::Mat blurredImage = cv::Mat::zeros(inputImage.size(), inputImage.type());

            // Record the starting time
            auto start = std::chrono::high_resolution_clock::now();

            rgbToHsv(inputImage, hsvImage);

            // Record the ending time
            auto end = std::chrono::high_resolution_clock::now();

            // Calculate the duration and add it to the vector
            durationsHSV.push_back(end - start);

            // Record the starting time
            start = std::chrono::high_resolution_clock::now();

            boxBlur(inputImage, blurredImage, 10);

            // Record the ending time
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

        std::string notifyHsvRuntime = "Average Runtime HSV With OpenCV, Picture " + std::to_string(i + 1) + ": " + std::to_string(average_duration) + " seconds\n";

        // Output the average duration
        std::cout << notifyHsvRuntime;
        myfile << notifyHsvRuntime;

        average_duration = total_duration_blur.count() / num_runs;

        std::string notifyBlurRuntime = "Average Runtime Blur With OpenCV, Picture " + std::to_string(i + 1) + ": " + std::to_string(average_duration) + " seconds\n";
        // Output the average duration
        std::cout << notifyBlurRuntime;
        myfile << notifyBlurRuntime;

        durationsHSV.clear();
        durationsBlur.clear();
        myfile.close();
    }
}

void OpenCVImageProcessing::execute(std::vector<std::string>& files, std::string& path) {
    for (int i = 0; i < files.size(); i++) {
        cv::Mat inputImage = cv::imread(path + files.at(i), cv::IMREAD_UNCHANGED);
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
        std::cout << "Finished processing image " << i + 1 << " with OpenCV." << std::endl;

        // Display the results
        cv::imshow("Original Image", inputImage);
        cv::imshow("HSV Image", hsvImage);
        cv::imshow("Blurred Original Image", blurImage);
        cv::imshow("Blurred HSV Image ", blurHSVImage);
        cv::waitKey(0);
        cv::destroyAllWindows();

        // Specify the folder path to save the images
        std::string folderPath = "Results OpenCV\\";

        // Create .jpg file from the result
        std::string numberingFile = std::to_string(i + 1);
        std::string hsvImageFile = numberingFile + ".hsvImage.jpg";
        std::string blurredImageFile = numberingFile + ".blurredImage.jpg";
        std::string blurredHSVImageFile = folderPath + numberingFile + ".blurredHSVImage.jpg";
        cv::imwrite(hsvImageFile, hsvImage);
        cv::imwrite(blurredImageFile, blurImage);
        cv::imwrite(blurredHSVImageFile, blurHSVImage);
    }
}