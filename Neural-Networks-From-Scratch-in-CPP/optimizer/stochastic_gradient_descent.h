#ifndef stochastic_gradient_descent
#define stochastic_gradient_descent

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

class Optimizer_SGD{
    
    public:
    double learning_rate, current_learning_rate, decay, momentum;
    long long iterations;
    
    Optimizer_SGD(double learning_rate = 1.0, double decay = 0.0, double momentum = 0.0){
        this->learning_rate = learning_rate;
        this->current_learning_rate = learning_rate;
        this->decay = decay;
        this->iterations = 0;
        this->momentum = momentum;
    }

    // Call once before any parameter updates
    // A commonly-used solution to keep initial updates large and explore various learning rates during
    // training is to implement a learning rate decay. Learning Rate gradually decreases with the number of epochs
    void pre_update_params(){
        if(this->decay)
            this->current_learning_rate = this->learning_rate * (1.0 / (1.0 + this->decay * this->iterations));
    }

    void update_params(Layer_Dense& layer){

        vector<vector<double>> weights_updates(layer.weights.size(), vector<double>(layer.weights[0].size(), 0)), 
                                    biases_updates(layer.biases.size(), vector<double>(layer.biases[0].size(), 0));

        if(this->momentum){ // SGD with momentum

            // Build weight updates with momentum - take previous
            // updates multiplied by retain factor and update with
            // current gradients
            for(int i = 0; i < layer.weights.size(); i++){
                for(int j = 0; j < layer.weights[0].size(); j++){
                    weights_updates[i][j] = this->momentum * layer.weights_momentums[i][j] - this->current_learning_rate * layer.dweights[i][j];
                    layer.weights_momentums[i][j] = weights_updates[i][j];
                }
            }
            
            // Build bias updates
            for(int i = 0; i < layer.biases[0].size(); i++){
                biases_updates[0][i] = this->momentum * layer.biases_momentums[0][i] - this->current_learning_rate * layer.dbiases[0][i];
                layer.biases_momentums[0][i] = biases_updates[0][i];
            }
        }
        else{
            
            // Updating parameters of the layers by subtracting the current params with their fraction of gradients (gradient * learning rate)
            // Subtracting points towards the minimum of the function.
            // Vanilla SGD updates (as before momentum update)
            for(int i = 0; i < weights_updates.size(); i++){
                for(int j = 0; j < weights_updates[0].size(); j++)
                    weights_updates[i][j] = -1 * this->current_learning_rate * layer.dweights[i][j];
            }

            for(int i = 0; i < biases_updates[0].size(); i++)
                biases_updates[0][i] = -1 * this->current_learning_rate * layer.dbiases[0][i];
        }

        // Update weights and biases using either
        // vanilla or momentum updates
        for(int i = 0; i < layer.weights.size(); i++){
            for(int j = 0; j < layer.weights[0].size(); j++)
                layer.weights[i][j] += weights_updates[i][j];
        }
        
        for(int i = 0; i < layer.biases[0].size(); i++)
            layer.biases[0][i] += biases_updates[0][i];
    }

    // Call once after any parameter updates
    void post_update_params(){
        this->iterations++;
    }
};

#endif