#ifndef Softmax
#define Softmax

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

#include "../math_operations/numpy_operations.h"

using namespace std;

// Theory:

// The ReLU activation function is unbounded and not normalized with other units and exclusive (each output is independent of the others)
// Softmax function takes unnormalized input and produces a normalized distribution of probabilities for our classes, 
// it returns confidence scores for each class and will add up to 1 and largest confidence score is consider to be the predicted class for given input
// S(i, j) = exp ^ z(i, j) / sum(1 to L) (exp ^ (z(i, l))) where i = curr sample , j, l = curr output in that sample

// Exponentiation serves multiple purposes. To calculate the probabilities, we need non-negative values, exponential converts 
// negative values to either zero if neg inf or positive, for 0 returns 1

// The exponential function is a monotonic function. This means that, with higher input values,
// outputs are also higher, so we won’t change the predicted class after applying it while making
// sure that we get non-negative values. It also adds stability to the result as the normalized
// exponentiation is more about the difference between numbers than their magnitudes.

// we will be normalizing by sum of neurons outputs for each sample, sample wise total will be exponentiated and will be in denominator 
// and numerator will be each neuron output for particular sample resulting all outputs will be normalized
// So Finally we will get each sample's each neuron's output wise distribution normalized to get the distribution within 0 to 1, and
// if each neuron represents one class this would be class wise probability distribution and we get the predicted output for each sample.

// There are two main pervasive challenges with neural networks: “dead neurons”
// and very large numbers (referred to as “exploding” values). The exponential
// function used in softmax activation is one of the sources of exploding values. 
// To prevent this we will be subtracting the largest value from list of input values. We would then change the output values
// to always be in a range from some negative value up to 0, With Softmax, thanks to the normalization, we can subtract
// any value from all of the inputs, and it will not change the output we can prove this: This is another useful property of the exponentiated and normalized function
// proof : e ^ x1 / e ^ (x1 + x2) = e ^ (x1 - x2) / e ^ (x1 + 0) where input = [x1, x2] , sub by max (suppose max = x2) then input = [x1 - x2, 0]
// But division of inputs by any number gives different answers due to nonlinearity nature of exponentiation

class Activation_Softmax{

    public:
    numpy np;
    vector<vector<double>> output, dinputs, inputs;

    void forward(vector<vector<double>>& layer_output){

        vector<double> sum_of_values(layer_output.size(), 0);
        vector<vector<double>> exp_values(layer_output.size(), vector<double>(layer_output[0].size(), 0)), 
                                probabilities(layer_output.size(), vector<double>(layer_output[0].size(), 0));
        this->inputs = layer_output;

        for(int i = 0; i < layer_output.size(); i++){
            auto max_iter = max_element(layer_output[i].begin(), layer_output[i].end());
            double maxi = *max_iter;
            for(int j = 0; j < layer_output[0].size(); j++){
                exp_values[i][j] = exp(layer_output[i][j] - maxi);
                sum_of_values[i] += exp_values[i][j];
            }
        }

        for(int i = 0; i < layer_output.size(); i++){
            for(int j = 0; j < layer_output[0].size(); j++){
                probabilities[i][j] = exp_values[i][j] / sum_of_values[i];
            }
        }

        this->output = probabilities;
    }

    // Backward pass of Softmax 
    // Gradient of softmax consists of jacobian which is formed because we need to take derivative of each sample's each input's with respect
    // all of the inputs, and there are two cases in gradient of softmax
    // Grad(softmax) = S(i, j) * Kronecker_delta_func(j, k) - S(i, j) * S(i, k), we get this by long procedure involving two cases when j == k and j != k.
    // where S(i, j) is a softmax func with ith sample and jth input and Kronecker delta function returns 1 if i == j Otherwise 0
    // so in 2nd component we need to take a derivative of jth sample with respect to each kth sample and for all j's in that it will result 
    // into a jacobian for each sample and to subtract the 1st component which will be only non zero when we take j's derivative with respect to itself that will be 'k' at some point
    // And for subtraction dimensions should be matched also at j == k only we will have a value at 1st component which matches with 2nd components indexes when we put the 1st component in diagonal matrix 
    // also which is required to subtract. So finally after subtraction we have each sample gradient of softmax output and will be mutliplied with the loss functions gradient 
    // to avoid the formation of 3D matrix and complete matrix operations within 2D matrix itself after multiplication the result will be stored as row in dinputs 
    // And after all samples product with loss gradient we will have directly the gradient of softmax with respect to output including the product with loss functions gradient

    void backward(vector<vector<double>>& dvalues){
        
        this->dinputs.resize(dvalues.size(), vector<double>(dvalues[0].size(), 0)); // Assign the same size as of dvalues (number of samples, number of output neurons)
        vector<vector<double>> jacobian_matrix(dvalues[0].size(), vector<double>(dvalues[0].size(), 0)), matrix_product(dvalues[0].size(), vector<double>(dvalues[0].size(), 0));
        for(int i = 0; i < dvalues.size(); i++){

            vector<vector<double>> single_dvalues = {dvalues[i]}, single_output = {this->output[i]}, single_outputT = np.T(single_output);
            single_dvalues = np.T(single_dvalues);
            jacobian_matrix = np.diagflat(single_outputT); // getting a diagonal matrix with diagonal elements from single_output as the first component of gradient
            matrix_product = np.dot(single_outputT, single_output); // For second component of gradient
            jacobian_matrix = np.sub(jacobian_matrix, matrix_product); // Sample wise gradient of softmax function
            // We are calculating sample wise gradient and directly multiplying sample wise so as to reduce the 2D matrix into 1D matrix which will be 
            // stored as a row in dinputs as samplewise gradients, if we use complete gradient instead of sample wise the softmax gradient will be of 3D which would be incompatible for 
            // out forward pass and also other matrix operations
            
            vector<vector<double>> vec = np.dot(jacobian_matrix, single_dvalues); // output's column vector
            vec = np.T(vec); // converting into row vector
            this->dinputs[i] = vec[0]; // storing sample gradients as rows            
        }
    }

    vector<double> predictions(){
        return np.argmax(output);
    }
};

#endif