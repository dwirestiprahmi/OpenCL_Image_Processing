#include <iostream>
#include <fstream>
#include <chrono>
#include "OpenCLImageProcessing.h"
#include "ImageProcessorInterface.h"
#include "CpuImageProcessing.h"
#include "OpenCVImageProcessing.h"

void evaluateRuntime(std::vector<std::string>& files, std::string& path) {
    
    OpenCLImageProcessing oclip;
    OpenCVImageProcessing ocvip;
    CpuImageProcessing cip;

    // Number of runs
    const int num_runs = 100;

    /*oclip.runtime(files, path, num_runs);
    ocvip.runtime(files, path, num_runs);*/
    cip.runtime(files, path, num_runs);
}

void executeDemo(std::vector<std::string>& files, std::string& path) {
    // show the demos that can be run
    int option = 0;

    while (true) {
        std::cout << "\nWhich Demo?:" << std::endl;
        std::cout << "1. CPU" << std::endl;
        std::cout << "2. OpenCL" << std::endl;
        std::cout << "3. OpenCV" << std::endl;
        std::cout << "4. Return to main" << std::endl;
        std::cout << "5. End Program" << std::endl;

        std::cout << "Input (number 1-4): ";
        std::cin >> option;
        switch (option) {
            case 1: {
                CpuImageProcessing cip;
                cip.execute(files, path);
                break;
            }
            case 2: {
                OpenCLImageProcessing oclip;
                oclip.execute(files, path);
                break;
            }
            case 3: {
                OpenCVImageProcessing ocvlip;
                ocvlip.execute(files, path);
                break;
            }
            case 4: {
                // Return to the main menu
                return;
            }
            case 5: {
                // End the program
                std::cout << "Exiting program..." << std::endl;
                exit(0);
            }
            default:{
                std::cout << "Invalid Input. Please try again." << std::endl;
                continue;
            }

        }
    }
}

int main()
{
    // initialize images that are going to be used
    std::string path = "images\\";
    std::vector<std::string> files =
    {
        "animal\\1.kitten_small.jpg",
        "animal\\2.kitten_medium.jpg", 
        "animal\\3.kitten_large.jpg", 
        "animal\\4.kitten_huge.jpg",
        "nature\\1.nature_small.jpeg", 
        "nature\\2.nature_medium.jpeg", 
        "nature\\3.nature_large.jpeg", 
        "nature\\4.nature_mega.jpeg",
    };

    // show options that can be run
    int option = 0;

    while (true) {
        std::cout << "\n***TASK NUMBER 9: Image Processing RGB->HSV Blur with OpenCL***" << std::endl;
        std::cout << "Select one of the numbers of the following options:" << std::endl;
        std::cout << "1. Execute Demo" << std::endl;
        std::cout << "2. Execute Runtime Evaluation" << std::endl;
        std::cout << "3. End Program" << std::endl;

        std::cout << "Input (number 1-3): ";
        std::cin >> option;   
    
        switch (option){
            case 1: {
                executeDemo(files, path); 
                break; 
            }
            case 2: {
                evaluateRuntime(files, path); 
                break; 
            }
            case 3: {
                std::cout << "Exiting Program..." << std::endl;
                return 0; 
            }
            default: {
                std::cout << "Invalid Input. Please try again." << std::endl;
                continue;
            }
        }
    }   
    
    return 0;
}
