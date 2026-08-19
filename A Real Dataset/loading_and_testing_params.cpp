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
// cmake -DSOURCE_FILE_ARG="loading_and_testing_params.cpp" -G "MinGW Makefiles" ..
// cmake --build . 
// ./MyProgram.exe

// While we can save and load model parameter values, we still need to define the model. It must be
// the exact configuration as the model that we’re importing parameters from

#if NN_HAS_OPENCV
int main(){

    vector<vector<vector<double>>> X_test, X;
    vector<vector<double>> X_test_reshaped;
    vector<double> y_test, temp, y;

    // Data Loading
    create_data_mnist(X, X_test, y, y_test, "../A Real Dataset/fashion_mnist_images/");

    for(int i = 0; i < X_test.size(); i++){
        temp.clear();
        for(int j = 0; j < X_test[0].size(); j++){
            for(int k = 0; k < X_test[0][0].size(); k++){
                X_test[i][j][k] = (X_test[i][j][k] - 127.5) / 127.5;    // scaling the test set as per train set
                temp.push_back(X_test[i][j][k]);    // converting 3D to 2D dataset as our model works only on 2D matrices
            }
        }
        X_test_reshaped.push_back(temp);
    }

    Layer_Dense dense1(X_test_reshaped[0].size(), 64);   // (784, 64)
    Activation_ReLU activation1, activation2;
    Layer_Dense dense2(64, 64);
    Layer_Dense dense3(64, 10);
    Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;
    Accuracy_Categorical accuracy;
    // We don't need optimizer as we are directly loading the parameters of the model we trained earlier
    
    vector<Layer_Dense*> layer_params;
    layer_params.push_back(&dense1);
    layer_params.push_back(&dense2);
    layer_params.push_back(&dense3);

    vector<tuple<vector<vector<double>>, vector<vector<double>>>> parameters;
    load_parameters(parameters, "../A Real Dataset/fashion_mnist.parms");
    set_parameters(layer_params, parameters);
    cout<<"parameters loaded successfully!\n";

    int batch_size = 128;
    int validation_steps = X_test.size() / batch_size;
    if(batch_size * validation_steps != X_test.size()) validation_steps++;

    loss_activation.loss_function.new_pass();
    accuracy.new_pass();

    vector<vector<double>> batch_X(batch_size, vector<double>(X_test_reshaped[0].size()));
    vector<double> batch_y(batch_size);
    double validation_accuracy, validation_loss;
    for(long long j = 0; j < validation_steps; j++){
        
        long long new_batch_start = batch_size * j;
        long long new_batch_end = std::min(
            batch_size * (j + 1),
            static_cast<long long>(X_test_reshaped.size())
        );

        long long current_batch_size = new_batch_end - new_batch_start;
        batch_X.resize(current_batch_size, vector<double>(X_test_reshaped[0].size()));
        batch_y.resize(current_batch_size);

        long long k = 0;
        for(long long start = new_batch_start; start < new_batch_end; start++){
            batch_X[k] = X_test_reshaped[start];
            batch_y[k++] = y_test[start];
        }

        dense1.forward(batch_X);
        activation1.forward(dense1.output);
        dense2.forward(activation1.output);
        activation2.forward(dense2.output);
        dense3.forward(activation2.output);

        loss_activation.forward(dense3.output, batch_y);
        accuracy.compare(loss_activation.output, batch_y);
        accuracy.calculate();
        validation_loss = loss_activation.loss_function.calculate_accumulated();
        validation_accuracy = accuracy.calculate_accumulated();
    }

    // Averaged out loss and accuracy from above validation steps
    cout<<"validation"<<", acc: "<<validation_accuracy<<", loss: "<<validation_loss<<"\n";

    return 0;
}

#else

int main() {
    std::cerr
        << "OpenCV is not installed; "
        << "this example is disabled.\n";

    return 0;
}

#endif