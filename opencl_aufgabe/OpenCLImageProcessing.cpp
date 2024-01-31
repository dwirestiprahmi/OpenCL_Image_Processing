
#include "OpenCLImageProcessing.h"
#include "OpenCVImageProcessing.h"

OpenCLImageProcessing::OpenCLImageProcessing() {
    cl_int status;

    // Get all platforms (drivers)
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // Get default device of the default platform
    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
    device = devices[0];

    // Create the context
    context = cl::Context(device);

    // Create queue to which we will push commands for the device
    commandQueue = cl::CommandQueue(context, device);

    // Read the kernel code file
    std::string image_kernel = read_kernel("image_kernel.cl");

    // A list of pairs <kernel-code, string length>
    cl::Program::Sources sources;
    sources.push_back({ image_kernel.c_str(), image_kernel.length() });

    // Create and build the kernel-code
    program = cl::Program(context, sources);

    status = program.build({ device });
    if (status != CL_SUCCESS) {
        std::cerr << "There were problems when building the kernel!" << std::endl;
        std::cerr << "Build Status: " 
            << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]) << std::endl;
        std::cerr << "Build Log:\t " 
            << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
        exit(1);
    }
    else {
        std::cout << "Build successful!" << std::endl << std::endl;
    }
    
}

OpenCLImageProcessing::~OpenCLImageProcessing() {}

std::string OpenCLImageProcessing::read_kernel(const char* filename) {
    std::ifstream kernelFile(filename);
    std::string content(
        (std::istreambuf_iterator<char>(kernelFile)),
        std::istreambuf_iterator<char>()
    );
    kernelFile.close();
    return content;
}

void OpenCLImageProcessing::rgbToHsv(const cv::Mat& input, cv::Mat& output) {

    // Define the required puffer size
    size_t bufferSize = input.total() * input.channels() * sizeof(uchar);

    // Allocate memory on device
    cl::Buffer gpuBuffer(context, CL_MEM_READ_ONLY, bufferSize);
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, bufferSize);

    // Create a kernel and specify its name
    cl::Kernel kernel(program, "rgbToHsv");

    int width = input.cols;
    int height = input.rows;
    int depth = input.channels();

    // Specify the arguments of kernel function
    kernel.setArg(0, gpuBuffer);
    kernel.setArg(1, outputBuffer);
    kernel.setArg(2, width);
    kernel.setArg(3, height);
    kernel.setArg(4, depth);

    // Copy data into the GPU
    commandQueue.enqueueWriteBuffer(gpuBuffer, CL_TRUE, 0, bufferSize, input.data);

    // Set the size of our kernels. For that, first, check what is permissible by our GPU:
    size_t max_work_group_size;
    device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_work_group_size);

    // Calculate the local work size based on the square root of the maximum work group size
    int localSize = static_cast<int>(std::floor(std::sqrtf(static_cast<float>(max_work_group_size))));
    size_t local_size[3]{ localSize, max_work_group_size / localSize, 1 };

    // Calculate the global work size to cover the entire image
    size_t global_size[3]{
      // Ensure that the global size is a multiple of the local size
      std::ceil(static_cast<float>(width) / localSize) * localSize,
      std::ceil(static_cast<float>(height) / localSize) * localSize,
      1
    };

    // Execute kernel
    commandQueue.enqueueNDRangeKernel(
        kernel, cl::NullRange, cl::NDRange(global_size[0], global_size[1], global_size[2]), 
        cl::NDRange(local_size[0], local_size[1], local_size[2]));

    // Read the results from the device memory back into the host memory
    commandQueue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, bufferSize, output.data);

    // Close the command queue
    commandQueue.finish();
}

void OpenCLImageProcessing::boxBlur(const cv::Mat& input, cv::Mat& output, int kernelSize) {

    // Define the required puffer size
    size_t bufferSize = input.total() * input.channels() * sizeof(uchar);

    // Allocate memory on device
    cl::Buffer gpuBuffer(context, CL_MEM_READ_ONLY, bufferSize);
    cl::Buffer outputBuffer(context, CL_MEM_WRITE_ONLY, bufferSize);

    // Create a kernel and specify its name
    cl::Kernel kernel(program, "blur");

    int width = input.cols;
    int height = input.rows;
    int depth = input.channels();

    // Specify the arguments of kernel function
    kernel.setArg(0, gpuBuffer);
    kernel.setArg(1, outputBuffer);
    kernel.setArg(2, width);
    kernel.setArg(3, height);
    kernel.setArg(4, depth);
    kernel.setArg(5, kernelSize);

    // Copy data into the GPU
    commandQueue.enqueueWriteBuffer(gpuBuffer, CL_TRUE, 0, bufferSize, input.data);

    // Set the size of our kernels. For that, first, check what is permissible by our GPU:
    size_t max_work_group_size;
    device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &max_work_group_size);

    int localSize = static_cast<int>(std::floor(std::sqrtf(static_cast<float>(max_work_group_size))));
    size_t local_size[3]{ localSize, max_work_group_size / localSize, 1 };

    size_t global_size[3]{
      std::ceil(static_cast<float>(width) / localSize) * localSize,
      std::ceil(static_cast<float>(height) / localSize) * localSize,
      1
    };

    // Execute kernel
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(global_size[0], global_size[1], global_size[2]),
        cl::NDRange(local_size[0], local_size[1], local_size[2]));

    // Read the results from the device memory back into the host memory
    commandQueue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, bufferSize, output.data);

    // Close the command queue
    commandQueue.finish();
}

void OpenCLImageProcessing::runtime(std::vector<std::string>& files, std::string& path, int num_runs) {
    // Vector to store durations
    std::vector<std::chrono::duration<double>> durationsHSV;
    std::vector<std::chrono::duration<double>> durationsBlur;

    for (int i = 0; i < files.size(); i++) {

        for (int j = 0; j < num_runs; ++j) {
            cv::Mat inputImage = cv::imread(path + files.at(i), cv::IMREAD_UNCHANGED);
            cv::Mat hsvImage = cv::Mat::zeros(inputImage.size(), inputImage.type());
            cv::Mat blurredImage = cv::Mat::zeros(inputImage.size(), inputImage.type());

            // Record the starting time
            auto startHSV = std::chrono::high_resolution_clock::now();

            rgbToHsv(inputImage, hsvImage);

            // Record the ending time
            auto endHSV = std::chrono::high_resolution_clock::now();

            // Calculate the duration and add it to the vector
            durationsHSV.push_back(endHSV - startHSV);
            
            // Record the starting time
            auto startBlur = std::chrono::high_resolution_clock::now();

            boxBlur(inputImage, blurredImage, 10);

            // Record the ending time
            auto endBlur = std::chrono::high_resolution_clock::now();
           
            // Calculate the duration and add it to the vector
            durationsBlur.push_back(endBlur - startBlur);
            
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

        std::string notifyHsvRuntime = "Average Runtime HSV With OpenCL, Picture " + std::to_string(i + 1) + ": " + std::to_string(average_duration) + " seconds\n";

        // Output the average duration
        std::cout << notifyHsvRuntime;
        myfile << notifyHsvRuntime;

        average_duration = total_duration_blur .count() / num_runs;

        std::string notifyBlurRuntime = "Average Runtime Blur With OpenCL, Picture " + std::to_string(i + 1) + ": " + std::to_string(average_duration) + " seconds\n";
        // Output the average duration
        std::cout << notifyBlurRuntime;
        myfile << notifyBlurRuntime;

        durationsHSV.clear();
        durationsBlur.clear();
        myfile.close();
    }
}

void OpenCLImageProcessing::execute(std::vector<std::string>&files, std::string & path) {
    // To access the image processing operation with OpenCV
    OpenCVImageProcessing ocvip;

    // Process all of the images that are included in the files parameter
    for (int i = 0; i < files.size(); i++) {
        // Variables for original, hsv, and blurred image
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
        std::cout << "Finished processing image " << i + 1 << " with OpenCL." << std::endl;

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
        std::string folderPath = "Results OpenCL\\";

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