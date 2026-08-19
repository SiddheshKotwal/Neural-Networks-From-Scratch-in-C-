#ifndef categorical_cross_entropy
#define categorical_cross_entropy

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

#include "loss_functions.h"

using namespace std;

class CategoricalCrossentropyLoss : public Loss{

    public:
    vector<vector<double>> dinputs;

    // Cross Entropy Loss
    vector<double> forward(vector<vector<double>>& y_pred, vector<double>& y_true){

        // Clip data to prevent division by 0
        // Clip both sides to not drag mean towards any value
        vector<vector<double>> y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7);
        vector<double> correct_confidences(y_pred.size(), 0), negative_log_likelihoods(y_pred.size(), 0);

        // Getting the confidence values from y_pred by mapping the indexes from y_true
        // only if categorical/sparse labels -> (1, 2, 3)
        for(int i = 0; i < y_pred.size(); i++)
            correct_confidences[i] = y_pred[i][y_true[i]];
            
        // Losses
        negative_log_likelihoods = np.negative_log(correct_confidences);
        return negative_log_likelihoods; // Predicted Sample Losses
    }

    // Cross Entropy Loss
    // Function overloaded for one-hot encoded vector 
    vector<double> forward(vector<vector<double>>& y_pred, vector<vector<double>>& y_true){

        // Clip data to prevent division by 0
        // Clip both sides to not drag mean towards any value
        vector<vector<double>> y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7);
        vector<double> correct_confidences(y_pred.size(), 0), negative_log_likelihoods(y_pred.size(), 0);

        // Mask values - only for one-hot encoded labels
        for(int i = 0; i < y_pred.size(); i++){
            for(int j = 0; j < y_pred[0].size(); j++)
                if(y_true[i][j] == 1) correct_confidences[i] = y_pred[i][j];
        }

        negative_log_likelihoods = np.negative_log(correct_confidences);
        return negative_log_likelihoods; // Predicted Sample losses
    }

    // Backward Pass for sparse labels
    void backward(vector<vector<double>>& dvalues, vector<double>& y_true){

        vector<vector<double>> one_hot(dvalues.size(), vector<double>(dvalues[0].size(), 0));
        this->dinputs.resize(dvalues.size(), vector<double>(dvalues[0].size(), 0)); // dim(number of samples, number of output neurons)
        // labels are sparse, turn them into one-hot vector
        for(int i = 0; i < one_hot.size(); i++)
            one_hot[i][y_true[i]] = 1;

        // Gradient of cross entropy loss function with respect to the y_hat
        for(int i = 0; i < one_hot.size(); i++)
            this->dinputs[i][y_true[i]] = -1 * (one_hot[i][y_true[i]] / dvalues[i][y_true[i]]);

        // Normalize the gradients by dividing by the number of samples to ensure that their magnitude 
        // is independent of the dataset size, making it easier to adjust the learning rate effectively.
        for(int i = 0; i < one_hot.size(); i++)
            this->dinputs[i][y_true[i]] = this->dinputs[i][y_true[i]] / dvalues.size();
    }
};

#endif