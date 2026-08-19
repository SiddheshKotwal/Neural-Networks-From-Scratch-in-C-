#ifndef dense_layer
#define dense_layer

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
#include <tuple>
#include <omp.h>

#include "../math_operations/numpy_operations.h"

using namespace std;

// Dense Layer (Layer in which all neurons from previous / input layer are connected with all the neurons of current layer (eg. like complete graph))
class Layer_Dense{
    
    public:
    numpy np;
    vector<vector<double>> weights, biases, output, inputs, dweights, dbiases, dinputs, weights_momentums, biases_momentums, weight_cache, bias_cache;
    double weight_regularizer_l1, weight_regularizer_l2, bias_regularizer_l1, bias_regularizer_l2;

    // Layer Initialization
    Layer_Dense(long long n_inputs, long long n_neurons, double weight_regularizer_l1 = 0, double weight_regularizer_l2 = 0, double bias_regularizer_l1 = 0, double bias_regularizer_l2 = 0){

        // initialize weights and biases dimensions according to the number of inputs and number of neurons in current layer
        // Here we have flipped the dimensions as (inputs, neurons) rather than (neurons, inputs) for weights which would reduce 
        // transpose operation on inputs matrix as we did in previous code
        this->weights.resize(n_inputs, vector<double>(n_neurons, 0));
        this->weights_momentums = this->weights; // Used for SGD with momentum
        this->weight_cache = this->weights; // Used for Adagrad
        // Set regularization strength
        this->weight_regularizer_l1 = weight_regularizer_l1;
        this->weight_regularizer_l2 = weight_regularizer_l2;
        this->bias_regularizer_l1 = bias_regularizer_l1;
        this->bias_regularizer_l2 = bias_regularizer_l2;

        // We generally set weights with random numbers not zeroes(because it would lead to dead network) but also not large and biases to zero
        // but sometimes we also would need to set bias to nonzero in cases where the activation function sets all inputs to zero
        this->biases.resize(1, vector<double>(n_neurons, 0));
        this->biases_momentums = this->biases; // Used for SGD with momentum
        this->bias_cache = this->biases; // Used for Adagrad

        // setting random weights 
        set_weights(weights);
    }

    void set_weights(vector<vector<double>>& weights){

        /*  Keras library uses this weights initializer
            Glorot uniform initializer, also called Xavier uniform initializer.
            
            It draws samples from a uniform distribution within [-limit, limit]
            where `limit` is `sqrt(6 / (fan_in + fan_out))`
            where `fan_in` is the number of input units in the weight tensor
            and `fan_out` is the number of output units in the weight tensor.
            
            # Arguments
            seed: A Python integer. Used to seed the random generator.
            
            # Returns
            An initializer.
            
            # References
            Glorot & Bengio, AISTATS 2010
            http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
        */

        // Generating random numbers from gaussian distribution of mean 0 and variance 1 -> (-1, 1)
        mt19937 rng(0); // seed 0
        normal_distribution<> dist(0, 1); // mean = 0, var = 1

        // setting random weights for the neurons by taking random number from gaussian distribution
        for(int i = 0; i < weights.size(); i++){
            for(int j = 0; j < weights[0].size(); j++){
                weights[i][j] = dist(rng) * 0.01;   // multiply with 0.1 for regression model and for classification 0.01
                // — the fraction (0.01) that multiplies the draw from the uniform distribution
                // depends on the number of inputs and the number of neurons and is not constant like in our case.
                // This method of initialization is called Glorot uniform.
                // Weight initialization process is very important it makes a model not learning at all into a learning state just by changing the multiplying factor to random weights.
            }
        }
    }

    void forward(vector<vector<double>>& inputs){
        
        // Used for backpropagation
        this->inputs = inputs;
        
        // setting the size of output and performing matrix multiplication and adding biases and calculating output
        this->output.resize(inputs.size(), vector<double>(weights[0].size(), 0));
        this->output = np.dot(inputs, this->weights);
        this->output = np.addBias(this->output, this->biases[0]);
    }

    void backward(vector<vector<double>>& dvalues){

        // Gradients on parameters
        vector<vector<double>> inputsT = np.T(this->inputs), weightsT = np.T(this->weights), regw_l1, regb_l1;
        regw_l1 = this->weights;
        regb_l1 = this->biases;

        // inputsT (number of neurons (n_inputs input layer), number of samples), dvalues (number of samples, number of neurons in layer)
        // dweights will be of same dimensions as weights of the layer
        this->dweights = np.dot(inputsT, dvalues); // gradient with respect to weights is just inputs
        this->dbiases = np.sum(dvalues, 0); // gradient with respect to biases is 1, we will sum all of the columns of dvalues, dbiases will have same dimensions as biases dimensions of that layer
        
        // Gradients on regularization
        // L1 on weights
        if(this->weight_regularizer_l1 > 0){
            for(int i = 0; i < regw_l1.size(); i++){
                for(int j = 0; j < regw_l1[0].size(); j++){
                    if(this->weights[i][j] < 0) regw_l1[i][j] = -1;
                    else regw_l1[i][j] = 1;
                }
            }
            
            for(int i = 0; i < regw_l1.size(); i++){
                for(int j = 0; j < regw_l1[0].size(); j++){
                    this->dweights[i][j] += this->weight_regularizer_l1 * regw_l1[i][j];
                }
            }
        }
        
        // L2 on weights
        if(this->weight_regularizer_l2 > 0){
            for(int i = 0; i < this->dweights.size(); i++){
                for(int j = 0; j < this->dweights[0].size(); j++)
                    this->dweights[i][j] += 2 * this->weight_regularizer_l2 * this->weights[i][j];
            }
        }

        // L1 on biases
        if(this->bias_regularizer_l1 > 0){
            for(int i = 0; i < regb_l1[0].size(); i++){
                if(this->biases[0][i] < 0) regb_l1[0][i] = -1;
                else regb_l1[0][i] = 1;
                this->dbiases[0][i] += this->bias_regularizer_l1 * regb_l1[0][i];
            }
        }

        // L2 on biases
        if(this->bias_regularizer_l2 > 0){
            for(int i = 0; i < this->dbiases[0].size(); i++)
                this->dbiases[0][i] += 2 * this->bias_regularizer_l2 * this->biases[0][i];
        }

        // Gradient on values
        this->dinputs = np.dot(dvalues, weightsT); // gradient with respect to inputs is just weights and 
        // we need this in multiple layers because these will be the outputs of previous layer whose gradients are required. 
        // Dimensions of dinputs will be same as of dimensions of inputs matrix to this layer or output matrix of previous layer
    }

    tuple<vector<vector<double>>, vector<vector<double>>> get_parameters(){
        return make_tuple(this->weights, this->biases);
    }

    void set_parameters(tuple<vector<vector<double>>, vector<vector<double>>>& parameters){
        tie(this->weights, this->biases) = parameters;
    }
};

#endif