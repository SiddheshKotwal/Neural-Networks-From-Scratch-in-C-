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

// Ensure that both training and testing data are scaled using identical methods. 
// Preprocessing rules should be derived solely from the training dataset. Any preprocessing rules 
// should be derived without knowledge of the testing dataset, but then applied to the testing set.
// Common mistake: Allowing the testing dataset to inform transformations made to the training dataset.
// Exceptions: Linear scaling by a constant is permissible (e.g., division by 255).
// Use min/max values or other methods like average and standard deviation for scaling (when we have extreme outliers)

// Note: 
// Testing data may not fit neatly within the scaled bounds of the training data. 
// In case where we are scaling by considering the values in training dataset/ only informing transformations based on training dataset 
// which is the correct method of scaling except when we use some constant to scale linearly in that case we scale train and test dataset by same constant

// Commands to run the project
// cmake -G "MinGW Makefiles" ..
// cmake --build .
// ./run_my_plot (incase of use of matplotlibcpp) or
// ./MyProgram.exe

#if NN_HAS_OPENCV
int main() {

    vector<vector<vector<double>>> X, X_test;
    vector<vector<double>> X_reshaped, X_test_reshaped, X_shuffled;
    vector<unsigned char> temp_2d;
    vector<double> y, y_test, temp, y_shuffled;

    // Data Loading
    create_data_mnist(X, X_test, y, y_test, "../A Real Dataset/fashion_mnist_images/");
    
    cout <<"\nTraining Set: " << X.size() << " " << X[0].size() << " " << X[0][0].size() << "\n";
    cout<<y.size()<<"\n";
    for(int i = 0; i < X.size(); i += 6000) cout<<y[i]<<"\n";
    cout << X_test.size() <<" " << X_test[0].size() <<" " << X_test[0][0].size()<<"\n";
    cout <<y_test.size()<<"\n";
    
    // Data Preprocessing
    // We are scaling within values -1 to 1
    for(int i = 0; i < X.size(); i++){
        temp.clear();
        for(int j = 0; j < X[0].size(); j++){
            for(int k = 0; k < X[0][0].size(); k++){
                X[i][j][k] = (X[i][j][k] - 127.5) / 127.5;
                // We need to reshape the dataset because our dense layer expects inputs as 2D Matrix and not 3D
                temp.push_back(X[i][j][k]);
            }
        }
        X_reshaped.push_back(temp);
    }

    for(int i = 0; i < X_test.size(); i++){
        temp.clear();
        for(int j = 0; j < X_test[0].size(); j++){
            for(int k = 0; k < X_test[0][0].size(); k++){
                X_test[i][j][k] = (X_test[i][j][k] - 127.5) / 127.5;
                temp.push_back(X_test[i][j][k]);
            }
        }
        X_test_reshaped.push_back(temp);
    }

    cout<<X_reshaped.size() <<" " << X_reshaped[0].size()<<"\n";
    cout<<X_test_reshaped.size() <<" " << X_test_reshaped[0].size()<<"\n";

    // There are neural network models called convolutional neural networks that will allow you
    // to pass 2D image data “as is,” but a dense neural network like we have here expects samples that
    // are 1D. Even in convolutional neural networks, you will usually need to flatten data before
    // feeding them to an output layer or a dense layer.

    // Data Shuffling

    // Ensure that the dataset is shuffled before training to prevent the model from learning spurious patterns 
    // due to the ordered arrangement of samples and their target classifications.
    // Training on unshuffled data can lead to:
    // 1. The model becoming biased towards predicting the same class within initial batches.
    // 2. Loss spikes and poor performance as the model encounters different classes in subsequent batches.
    // 3. Difficulty in finding a global minimum due to cycling between local minimums for each class.
    // Shuffling the data ensures that each batch contains a mix of different classes, promoting better generalization.
    // When shuffling, ensure that both the samples and their corresponding targets are shuffled in unison.
    // We didn't shuffle the dataset in our previous models because it was small dataset and also we were training by using 
    // all samples in single time but in this case we will be training in batches because the dataset is large.
    // Shuffling a dataset is generally a good practice in machine learning and neural network training.

    temp.clear();
    for(int i = 0; i < y.size(); i++) temp.push_back(i);

    mt19937 rand(0);
    shuffle(temp.begin(), temp.end(), rand);  // shuffling indices to further shuffle X(samples) and y(labels) in unison.

    for(int i = 0; i < temp.size(); i++){
        X_shuffled.push_back(X_reshaped[temp[i]]);
        y_shuffled.push_back(y[temp[i]]);
    }

    // If the model does not train or appears to be misbehaving, you will want to double-check how you preprocessed the data.
    // check if the shuffling is correct or not
    for(int i = 0; i < X_shuffled[0].size(); i++) temp_2d.push_back(static_cast<unsigned char>(round((X_shuffled[4][i] * 127.5) + 127.5)));
    plt::imshow(temp_2d.data(), 28, 28, 1, {{"cmap", "gray"}}); // The image might be looking slightly different because we scaled the data and then shuffled it.
    plt::show();
    cout<<"label after shuffling: "<< y[4]<<"\n";
    // Actually first shuffling then scaling of dataset is common

    // Batches
    // Common batch sizes range between 32 and 128 samples. Traning large batch sizes will become slow compared to the speed
    // achievable with smaller batch sizes. Each batch of samples being trained is referred to as a step. We can calculate the number
    // of steps by dividing the number of samples by the batch size

    // This is the number of iterations that we’ll be making per epoch in a loop. If there are some straggler samples left over, 
    // we can add them in by simply adding one more step. We will be calculating loss and accuracy for each step and epoch.

    int batch_size = 128;
    int train_steps = X.size() / batch_size;
    if(batch_size * train_steps != X.size()) train_steps++;

    int validation_steps = X_test.size() / batch_size;
    if(batch_size * validation_steps != X_test.size()) validation_steps++;
    long long epoch = 10;

    Layer_Dense dense1(X_shuffled[0].size(), 64);   // (784, 64)
    Activation_ReLU activation1, activation2;
    Layer_Dense dense2(64, 64);
    Layer_Dense dense3(64, 10);
    Activation_Softmax_Loss_CategoricalCrossentropy loss_activation;
    Accuracy_Categorical accuracy;
    Optimizer_Adam optimizer(0.001, 1e-3);  // (0.001, 1e-3)

    vector<vector<double>> batch_X(batch_size, vector<double>(X_shuffled[0].size()));
    vector<double> batch_y(batch_size);
    for(long long i = 0; i < epoch; i++){

        cout<<"epoch "<<i + 1<<":"<<"\n";
        loss_activation.loss_function.new_pass();
        accuracy.new_pass();

        for(long long j = 0; j < train_steps; j++){
            
            long long k = 0, new_batch_start = batch_size * j, new_batch_end = batch_size * (j + 1);
            for(long long start = new_batch_start; start < new_batch_end; start++){
                batch_X[k] = X_shuffled[start];
                batch_y[k++] = y_shuffled[start];
            }

            dense1.forward(batch_X);
            activation1.forward(dense1.output);
            dense2.forward(activation1.output);
            activation2.forward(dense2.output);
            dense3.forward(activation2.output);
            double data_loss = loss_activation.forward(dense3.output, batch_y);
            double reg_loss = loss_activation.loss_function.regularization_loss(dense1) + loss_activation.loss_function.regularization_loss(dense2) + loss_activation.loss_function.regularization_loss(dense3);
            accuracy.compare(loss_activation.output, batch_y);
            double accuracy_ = accuracy.calculate();
            double loss = data_loss + reg_loss;

            if(!(j % 100) || j + 1 == train_steps) cout<<"step: "<<j<<", acc: "<<accuracy_<<", loss: "<<loss<<", data_loss: "<<data_loss<<", reg_loss: "<<reg_loss<<", lr: "<<optimizer.current_learning_rate<<"\n";

            loss_activation.backward(loss_activation.output, batch_y);
            dense3.backward(loss_activation.dinputs);
            activation2.backward(dense3.dinputs);
            dense2.backward(activation2.dinputs);
            activation1.backward(dense2.dinputs);
            dense1.backward(activation1.dinputs);

            // Update weights and biases
            optimizer.pre_update_params();
            optimizer.update_params(dense1);
            optimizer.update_params(dense2);
            optimizer.update_params(dense3);
            optimizer.post_update_params();
        }

        double epoch_data_loss = loss_activation.loss_function.calculate_accumulated();
        double epoch_reg_loss = loss_activation.loss_function.regularization_loss(dense1) + loss_activation.loss_function.regularization_loss(dense2) + loss_activation.loss_function.regularization_loss(dense3);
        double epoch_loss = epoch_data_loss + epoch_reg_loss;
        double epoch_accuracy = accuracy.calculate_accumulated();

        // Averaged out loss and accuracy from above training steps
        cout<<"training"<<", acc: "<<epoch_accuracy<<", loss: "<<epoch_loss<<", data_loss: "<<epoch_data_loss<<", reg_loss: "<<epoch_reg_loss<<", lr: "<<optimizer.current_learning_rate<<"\n";

        // Validation 
        loss_activation.loss_function.new_pass();
        accuracy.new_pass();

        double validation_accuracy, validation_loss;
        for(long long j = 0; j < validation_steps; j++){
            
            long long k = 0, new_batch_start = batch_size * j, new_batch_end = batch_size * (j + 1);
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

        // If we have out of sample data we can run our validation code on it to evaluate our model's final loss and accuracy,
        // as we have used test set for validation we don't have dataset for evaluation.
        // We will often train a model tweak it's hyperparameters, train it all over again and so on using 
        // training and validation data. Then, whenever we find the model and hyperparameters that appear 
        // to perform the best, we’ll use that model on testing data and, in the future, to make predictions in production.
    }

    // Saving the Parameters
    vector<Layer_Dense> layer_params;
    layer_params.push_back(dense1);
    layer_params.push_back(dense2);
    layer_params.push_back(dense3);

    vector<tuple<vector<vector<double>>, vector<vector<double>>>> parameters = get_parameters(layer_params);
    save_parameters(parameters, "../A Real Dataset/fashion_mnist.parms");
    cout<<"parameters saved successfully!\n";

    // With saved weights, we can, for example, initialize a model with those weights, trained from similar data, and then train that
    // model to work with our specific data. This is called transfer learning. Weights can be used to visualize the model, 
    // identify dead neurons, implement more complicated models (like reinforcement learning, where weights collected
    // from multiple models are committed to a single network)

    // Saving the model
    // We need to create a model class to save it but as we have not created model class we can't save it completely and
    // saving individual subclasses of model becomes so complicated. So, we will be just implementing the saving and loading of parameters.
    // Saving model should consist of only important properties except the object attributes such as inputs, output, dinputs, dweights,
    // dbiases for all classes. We just need to save the data which we require to start the optimization process again from previous
    // checkpoint and the metrics to evaluate and monitor the training process continuing from previous checkpoint like loss, accuracy, etc.

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