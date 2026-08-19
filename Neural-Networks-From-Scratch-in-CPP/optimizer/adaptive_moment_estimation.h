#ifndef adaptive_moment_estimation
#define adaptive_moment_estimation

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

class Optimizer_Adam{
    
    public:
    double learning_rate, current_learning_rate, decay, epsilon, beta1, beta2;
    long long iterations;
    
    // Initialize optimizer - set settings
    Optimizer_Adam(double learning_rate = 0.001, double decay = 0.0, double epsilon = 1e-7, double beta1 = 0.9, double beta2 = 0.999){
        this->learning_rate = learning_rate;
        this->current_learning_rate = learning_rate;
        this->decay = decay;
        this->iterations = 0;
        this->epsilon = epsilon;
        this->beta1 = beta1;
        this->beta2 = beta2;
    }

    // Call once before any parameter updates
    // A commonly-used solution to keep initial updates large and explore various learning rates during
    // training is to implement a learning rate decay. Learning Rate gradually decreases with the number of epochs
    void pre_update_params(){
        if(this->decay)
            this->current_learning_rate = this->learning_rate * (1.0 / (1.0 + this->decay * this->iterations));
    }

    void update_params(Layer_Dense& layer){

        // Update momentum with current gradients
        for(int i = 0; i < layer.weights.size(); i++){
            for(int j = 0; j < layer.weights[0].size(); j++){
                layer.weights_momentums[i][j] = this->beta1 * layer.weights_momentums[i][j] + (1 - this->beta1) * layer.dweights[i][j];
            }
        }
        
        for(int i = 0; i < layer.biases[0].size(); i++)
            layer.biases_momentums[0][i] = this->beta1 * layer.biases_momentums[0][i] + (1 - this->beta1) * layer.dbiases[0][i];

        // Get corrected momentum
        // iteration is 0 at first pass
        // and we need to start with 1 here
        vector<vector<double>> weight_momentums_corrected(layer.weights.size(), vector<double>(layer.weights[0].size(), 0)),
                                weight_cache_corrected(layer.weights.size(), vector<double>(layer.weights[0].size(), 0)), 
                                bias_momentums_corrected(layer.biases.size(), vector<double>(layer.biases[0].size(), 0)),
                                bias_cache_corrected(layer.biases.size(), vector<double>(layer.biases[0].size(), 0));
        
        for(int i = 0; i < layer.weights.size(); i++){
            for(int j = 0; j < layer.weights[0].size(); j++){
                weight_momentums_corrected[i][j] = layer.weights_momentums[i][j] / (1 - pow(this->beta1, this->iterations + 1));
            }
        }

        for(int i = 0; i < layer.biases[0].size(); i++)
            bias_momentums_corrected[0][i] = layer.biases_momentums[0][i] / (1 - pow(this->beta1, this->iterations + 1));

        // Update cache with squared current gradients
        for(int i = 0; i < layer.weights.size(); i++){
            for(int j = 0; j < layer.weights[0].size(); j++){
                layer.weight_cache[i][j] = this->beta2 * layer.weight_cache[i][j] + (1 - this->beta2) * layer.dweights[i][j] * layer.dweights[i][j];
            }
        }

        for(int i = 0; i < layer.biases[0].size(); i++)
            layer.bias_cache[0][i] = this->beta2 * layer.bias_cache[0][i] + (1 - this->beta2) * layer.dbiases[0][i] * layer.dbiases[0][i];

        // Get corrected cache
        for(int i = 0; i < layer.weights.size(); i++){
            for(int j = 0; j < layer.weights[0].size(); j++){
                weight_cache_corrected[i][j] = layer.weight_cache[i][j] / (1 - pow(this->beta2, this->iterations + 1));
            }
        }

        for(int i = 0; i < layer.biases[0].size(); i++)
            bias_cache_corrected[0][i] = layer.bias_cache[0][i] / (1 - pow(this->beta2, this->iterations + 1));

        // Vanilla SGD parameter update + normalization
        // with square rooted cache
        for(int i = 0; i < layer.weights.size(); i++){
            for(int j = 0; j < layer.weights[0].size(); j++)
                layer.weights[i][j] += -1 * this->current_learning_rate * weight_momentums_corrected[i][j] / (sqrt(weight_cache_corrected[i][j]) + this->epsilon);
        }
        
        for(int i = 0; i < layer.biases[0].size(); i++)
            layer.biases[0][i] += -1 * this->current_learning_rate * bias_momentums_corrected[0][i] / (sqrt(bias_cache_corrected[0][i]) + this->epsilon);
    }

    // Call once after any parameter updates
    void post_update_params(){
        this->iterations++;
    }
};

#endif