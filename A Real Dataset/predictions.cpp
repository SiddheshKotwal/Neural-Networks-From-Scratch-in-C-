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

#include "get_dataset.cpp"
#include "get_set_params.cpp"

// Commands to run the project
// cmake -DSOURCE_FILE_ARG="predictions.cpp" -G "MinGW Makefiles" ..
// cmake --build .
// ./run_my_plot (incase of use of matplotlibcpp) or
// ./MyProgram.exe

#if NN_HAS_OPENCV
int main(){

    /*  // Predicting on test data
        vector<vector<vector<double>>> X_test, X;
        vector<double> y_test, y;

        // Data Loading
        create_data_mnist(X, X_test, y, y_test, "../A Real Dataset/fashion_mnist_images/");
    */

    // To make predictions we need to convert the image data into data containing same properties as the image dataset used for training the model,
    // In our case training dataset was in grayscale, with 28 x 28 resolution image and we also scaled and flatten the image data before training. 
    // So, we need to convert our data into grayscale and 28 x 28 resolution image and then scale the image data b/w -1 to 1 and flatten into 784 element array.
    // And make a vector of (number of samples, 784) for predictions.
    
    cv::Mat image = cv::imread("../A Real Dataset/tshirt.png", cv::IMREAD_GRAYSCALE);
    cv::resize(image, image, cv::Size(28, 28));
    plt::imshow(image.data, image.rows, image.cols, 1, {{"cmap", "gray"}});
    plt::show();

    vector<vector<vector<double>>> image_data;
    vector<vector<double>> matrix(28, vector<double>(28, 0)), image_data_reshaped, output;
    vector<double> temp, image_test;
    image_test.push_back(0); // As we know the image is shirt label 0

    #pragma omp parallel for
    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++)
            matrix[i][j] = 255 - static_cast<double>(image.at<uchar>(i, j));
            // We are color-inverting the image for predictions because our training set images are color-inverted
            // changing black with white and so on by subtracting pixel value from 255. Without color-inversion we would get wrong predictions.
    }
    image_data.push_back(matrix);

    /*
        // Predicting on test data of 5 images
        for(int i = 0; i < 5; i++){
            image_data.push_back(X_test[i]);
            image_test.push_back(y_test[i]);
        }
    */

    for(int i = 0; i < image_data.size(); i++){
        temp.clear();
        for(int j = 0; j < image_data[0].size(); j++){
            for(int k = 0; k < image_data[0][0].size(); k++){
                image_data[i][j][k] = (image_data[i][j][k] - 127.5) / 127.5;    // scaling the test set as per train set
                temp.push_back(image_data[i][j][k]);    // converting 3D to 2D dataset as our model works only on 2D matrices
            }
        }
        image_data_reshaped.push_back(temp);
    }

    Layer_Dense dense1(image_data_reshaped[0].size(), 64);   // (784, 64)
    Activation_ReLU activation1, activation2;
    Layer_Dense dense2(64, 64);
    Layer_Dense dense3(64, 10);
    Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;
    Accuracy_Categorical accuracy;

    vector<Layer_Dense*> layer_params;
    layer_params.push_back(&dense1);
    layer_params.push_back(&dense2);
    layer_params.push_back(&dense3);

    vector<tuple<vector<vector<double>>, vector<vector<double>>>> parameters;
    load_parameters(parameters, "../A Real Dataset/fashion_mnist.parms");
    set_parameters(layer_params, parameters);
    cout<<"parameters loaded successfully!\n";

    // We can do batch wise predictions too 
    int batch_size = 1;
    int prediction_steps = image_data_reshaped.size() / batch_size;
    if(batch_size * prediction_steps != image_data_reshaped.size()) prediction_steps++;

    loss_activation.loss_function.new_pass();
    accuracy.new_pass();

    vector<vector<double>> batch_X(batch_size, vector<double>(image_data_reshaped[0].size()));
    vector<double> batch_y(batch_size);
    double predictions_accuracy, predictions_loss;
    for(long long j = 0; j < prediction_steps; j++){
        
        long long k = 0, new_batch_start = batch_size * j, new_batch_end = batch_size * (j + 1);
        for(long long start = new_batch_start; start < new_batch_end; start++){
            batch_X[k] = image_data_reshaped[start];
            batch_y[k++] = image_test[start];
        }

        dense1.forward(batch_X);
        activation1.forward(dense1.output);
        dense2.forward(activation1.output);
        activation2.forward(dense2.output);
        dense3.forward(activation2.output);

        loss_activation.forward(dense3.output, batch_y);
        output.push_back(loss_activation.activation.predictions());
        accuracy.compare(loss_activation.output, batch_y);
        accuracy.calculate();
        predictions_loss = loss_activation.loss_function.calculate_accumulated();
        predictions_accuracy = accuracy.calculate_accumulated();
    }

    // Averaged out loss and accuracy from above validation steps
    cout<<"predictions"<<", acc: "<<predictions_accuracy<<", loss: "<<predictions_loss<<"\n";

    map<double, string> fashion_mnist_labels = {
        {0, "T-shirt/top"},
        {1, "Trouser"},
        {2, "Pullover"},
        {3, "Dress"},
        {4, "Coat"},
        {5, "Sandal"},
        {6, "Shirt"},
        {7, "Sneaker"},
        {8, "Bag"},
        {9, "Ankle boot"}
    };

    for(int i = 0; i < output.size(); i++){
        cout<<"Batch "<<(i + 1)<<": \n";
        for(int j = 0; j < output[0].size(); j++)
            cout<<fashion_mnist_labels[output[i][j]]<<"\n";
    }

    // Now it works after color-inversion! The reason it works now, and not work previously, is from how the Dense layers
    // work — they learn feature (pixel in this case) values and the correlation between them. Contrast
    // this with convolutional layers, which are being trained to find and understand features on images
    // (not features as data input nodes, but actual characteristics/traits, such as lines and curves).
    // Because pixel values were very different, the model incorrectly put its “guess” in this case.
    // Convolutional layers may properly predict in this case, as-is.
}

#else

int main() {
    std::cerr
        << "OpenCV is not installed; "
        << "this example is disabled.\n";

    return 0;
}

#endif