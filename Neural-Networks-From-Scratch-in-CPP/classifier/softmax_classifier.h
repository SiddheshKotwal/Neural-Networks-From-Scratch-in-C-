#ifndef softmax_classifier
#define softmax_classifier

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

#include "../activation_functions/Softmax.h"
#include "../loss_functions/categorical_cross_entropy.h"

using namespace std;

class Activation_Softmax_Loss_CategoricalCrossentropy{
    
    public:
    Activation_Softmax activation;
    CategoricalCrossentropyLoss loss_function;
    vector<vector<double>> output, dinputs;

    double forward(vector<vector<double>>& inputs, vector<double>& y_true){

        // Output layer's activation function
        activation.forward(inputs);
        
        // Set the output
        this->output = activation.output; // storing to further calculate the backward gradient of loss function with respect to this softmax output

        // Calculate and return loss value
        vector<double> pred_losses = loss_function.forward(this->output, y_true);
        return loss_function.calculate(pred_losses);
    }

    void backward(vector<vector<double>>& dvalues, vector<double>& y_true){

        // Combined gradient results into just calculation of y_hat(i, k) - y(i, k) getting 7 times faster in python, instead of separate 
        // gradient of loss function with respect to it's inputs and gradient of softmax with repsect to it's inputs and then taking product of both.
        this->dinputs = dvalues; // copying to just edit it at ith sample's kth index by subtracting 1 to get final gradient
        for(int i = 0; i < dvalues.size(); i++){
            this->dinputs[i][y_true[i]] -= 1; // subtracting 1 because y(i, k) at k we have one and we want to subtract from y_hat's ith sample and kth class/neuron 
            // and we have kth class directly in y_true's ith index as classes are sparse not one-hot encoded.
        }

        // Normalize the gradients by dividing by the number of samples to ensure that their magnitude 
        // is independent of the dataset size, making it easier to adjust the learning rate effectively.
        for(int i = 0; i < this->dinputs.size(); i++){
            for(int j = 0; j < this->dinputs[0].size(); j++){
                this->dinputs[i][j] /= dvalues.size();
            }
        }
    }
};

#endif