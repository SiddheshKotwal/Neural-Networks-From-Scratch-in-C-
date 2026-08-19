#ifndef loss_functions
#define loss_functions

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <omp.h>

#include "../Layers/dense_layer.h"

using namespace std;

// Theory:

// We are Classifying in this example so we need different loss/cost function than the MSE used in (Linear regression).

// Categorical cross-entropy is explicitly used to compare a “ground-truth” probability (y or “targets”) 
// and some predicted distribution (y-hat or “predictions”). It is also one of the most
// commonly used loss functions with a softmax activation on the output layer.

// Formula : L(i) = - sum(j) (y(i, j) * log(y_hat(i, j))) where i = sample, j = output index, y = target value, y_hat = predicted value
// one-hot encoding (categorical values where only one target value is hot (on/1) and all other values are off(zero))
// we can simplify the formula as : L(i) = - log(y_hat(i, k)) where k = index of target label in case of one-hot encoding of target labels
// eg. output of one sample softmax_output = [0.22, 0.6, 0.18], target_output = [0, 1, 0] So here we have one-hot encoded target label
// with correct class as 2nd so the negative log of probability of the 2nd (index 1) output value would be taken as a loss for that sample.

// cross entropy loss -> multiple classes, log loss -> generally with binary classes 
// Here, we have multiple classes, cross entropy compares two probability distributions target values and predicted values
// higher the output values lower the log of the values representing the loss of neural network for that sample
// The natural log represents the solution for the x-term in the equation e^x= b where x = log(b).
// We will be calculating the loss of each sample in a batch and then averaging it for each batch

// Target labels could be represented as one-hot encoding or sparse (eg. for 3 classes):
// one-hot: [0, 1, 0] 
// sparse: 1, 2, 3 (direct class numbers)
// So finding the correct index in the output vector according to the target labels would be done in two ways for above 2 types of label representations in our dataset
// In our example we are using the sparse representation by spiral data function. We can detect the two ways by checking the vector dimensions 
// if dimensions are 2 then it's one-hot encoded and if it's 1 then it's sparse representation.

// The process is we need to get the probabilities of the output at index representing true target label and then using neg log.
// Lastly the problem is If the model assigns full confidence to a value
// that wasn’t the target. If we then try to calculate the loss of this confidence of 0. It will give undefined answer, to solve this issue we
// some very small value 1e-7 to the output, but this also causes problem if the prediction is 1 -> log(1 + 1e-7) gives negative value,
// So the solution is we clip some very small value 1e-7 from both ends and convert 0 to 1e-7 and 1 to 1 - 1e-7. So, the prediction values
// range in (1e-7, 1 - 1e-7)

// In each loss function we normalize the gradient in backward pass we normalize gradients by the number of samples to make them invariant to 
// the batch size, or the number of samples in general

// Common loss class
class Loss{

    public:
    numpy np;
    double accumulated_sum, accumulated_count;

    // Calculate mean Losses
    double calculate(vector<double>& sample_losses){

        double data_loss = np.mean(sample_losses);
        for(int i = 0; i < sample_losses.size(); i++)
            this->accumulated_sum += sample_losses[i];
        this->accumulated_count += sample_losses.size();
        return data_loss;
    }

    double regularization_loss(Layer_Dense &layer){
        
        // 0 by default
        double regularization_loss = 0.0;

        // L1 regularization - weights
        // calculate only when factor greater than 0
        if(layer.weight_regularizer_l1 > 0){
            double total = 0;
            for(int i = 0; i < layer.weights.size(); i++){
                for(int j = 0; j < layer.weights[0].size(); j++)
                    total += abs(layer.weights[i][j]);
            }
            regularization_loss += layer.weight_regularizer_l1 * total;
        }

        // L2 regularization - weights
        if(layer.weight_regularizer_l2 > 0){
            double total = 0;
            for(int i = 0; i < layer.weights.size(); i++){
                for(int j = 0; j < layer.weights[0].size(); j++)
                    total += layer.weights[i][j] * layer.weights[i][j];
            }
            regularization_loss += layer.weight_regularizer_l2 * total;
        }

        // L1 regularization - biases
        // calculate only when factor greater than 0
        if(layer.bias_regularizer_l1 > 0){
            double total = 0;
            for(int i = 0; i < layer.biases[0].size(); i++)
                total += abs(layer.biases[0][i]);
            regularization_loss += layer.bias_regularizer_l1 * total;
        }

        // L2 regularization - biases
        if(layer.bias_regularizer_l2 > 0){
            double total = 0;
            for(int i = 0; i < layer.biases[0].size(); i++)
                total += layer.biases[0][i] * layer.biases[0][i];
            regularization_loss += layer.bias_regularizer_l2 * total;
        }

        return regularization_loss;
    }

    // sample-wise average for overall loss
    double calculate_accumulated(){
        return this->accumulated_sum / this->accumulated_count;
    }

    // Reset variables for accumulated loss
    void new_pass(){
        this->accumulated_count = this->accumulated_sum = 0;
    }
};

#endif