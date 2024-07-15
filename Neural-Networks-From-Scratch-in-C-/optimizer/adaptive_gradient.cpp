#include<cmath>

class Optimizer_Adagrad{
    
    public:
    double learning_rate, current_learning_rate, decay, epsilon;
    long long iterations;
    
    // Initialize optimizer - set settings
    Optimizer_Adagrad(double learning_rate = 1.0, double decay = 0.0, double epsilon = 1e-7){
        this->learning_rate = learning_rate;
        this->current_learning_rate = learning_rate;
        this->decay = decay;
        this->iterations = 0;
        this->epsilon = epsilon;
    }

    // Call once before any parameter updates
    // A commonly-used solution to keep initial updates large and explore various learning rates during
    // training is to implement a learning rate decay. Learning Rate gradually decreases with the number of epochs
    void pre_update_params(){
        if(this->decay)
            this->current_learning_rate = this->learning_rate * (1.0 / (1.0 + this->decay * this->iterations));
    }

    void update_params(Layer_Dense& layer){

        // Update cache with squared current gradients
        for(int i = 0; i < layer.weights.size(); i++){
            for(int j = 0; j < layer.weights[0].size(); j++){
                layer.weight_cache[i][j] += layer.dweights[i][j] * layer.dweights[i][j];
            }
        }

        for(int i = 0; i < layer.biases[0].size(); i++)
            layer.bias_cache[0][i] += layer.dbiases[0][i] * layer.dbiases[0][i];

        // Vanilla SGD parameter update + normalization
        // with square rooted cache
        for(int i = 0; i < layer.weights.size(); i++){
            for(int j = 0; j < layer.weights[0].size(); j++)
                layer.weights[i][j] += -1 * this->current_learning_rate * layer.dweights[i][j] / (sqrt(layer.weight_cache[i][j]) + this->epsilon);
        }
        
        for(int i = 0; i < layer.biases[0].size(); i++)
            layer.biases[0][i] += -1 * this->current_learning_rate * layer.dbiases[0][i] / (sqrt(layer.bias_cache[0][i]) + this->epsilon);
    }

    // Call once after any parameter updates
    void post_update_params(){
        this->iterations++;
    }
};