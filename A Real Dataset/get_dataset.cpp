#include "../Neural-Networks-From-Scratch-in-CPP/common_includes.h"

#if defined(__has_include)
#  if __has_include(<opencv2/opencv.hpp>)
#    define NN_HAS_OPENCV 1
#    include <opencv2/opencv.hpp>
#  endif
#endif

#ifndef NN_HAS_OPENCV
#  define NN_HAS_OPENCV 0
#endif

#if NN_HAS_OPENCV
#include <filesystem>

namespace fs = std::filesystem;

void load_mnist_dataset(vector<vector<vector<double>>>& X, vector<double>& y, string dataset, string path){

    string directoryPath = path + dataset;

    try {
        for (auto& entry : fs::directory_iterator(directoryPath)) {
            if (fs::is_directory(entry)) {
                double label = stod(entry.path().filename().string());
                cout<<(int)label<<" ";
                for(auto& file_entry : fs::directory_iterator(entry.path())) {
                    if (fs::is_regular_file(file_entry)) {
                        cv::Mat img = cv::imread(file_entry.path().string(), cv::IMREAD_UNCHANGED);
                        vector<vector<double>> matrix(img.rows, vector<double>(img.cols, 0));
                        #pragma omp parallel for
                        for(int i = 0; i < img.rows; i++){
                            for(int j = 0; j < img.cols; j++)
                                matrix[i][j] = static_cast<double>(img.at<uchar>(i, j));
                        }

                        #pragma omp critical
                        {
                            X.push_back(matrix);
                            y.push_back(label);
                        }
                    }
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "General error: " << e.what() << std::endl;
    }
    cout<<"\n";
}

void create_data_mnist(vector<vector<vector<double>>>& X, vector<vector<vector<double>>>& X_test, vector<double>& y, vector<double>& y_test, string path){
    load_mnist_dataset(X, y, "train", path);
    load_mnist_dataset(X_test, y_test, "test", path);
}

#else

void load_mnist_dataset(
    vector<vector<vector<double>>>&,
    vector<double>&,
    string,
    string
) {
    throw runtime_error(
        "OpenCV is required to load the image dataset."
    );
}

void create_data_mnist(
    vector<vector<vector<double>>>&,
    vector<vector<vector<double>>>&,
    vector<double>&,
    vector<double>&,
    string
) {
    throw runtime_error(
        "OpenCV is required to load the image dataset."
    );
}

#endif
