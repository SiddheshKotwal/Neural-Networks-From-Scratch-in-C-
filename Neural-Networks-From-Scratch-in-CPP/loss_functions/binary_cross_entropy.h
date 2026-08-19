#ifndef binary_cross_entropy
#define binary_cross_entropy

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

class Loss_BinaryCrossentropy : public Loss{
    public:
    vector<vector<double>> dinputs;

    vector<double> forward(vector<vector<double>>& y_pred, vector<vector<double>>& y_true){

        // We have y_true here as matrix where each column represents the each neurons true class out of binary class (0 or 1)
        // And we can have mutliple neurons in output layer each neuron predicting the class out of binary class 
        // ex. 1 neurons can predict between dog or cat and 2nd neuron can predict between black or white, so each neuron predicts between it's distinct binary class

        // Clip data to prevent division by 0
        // Clip both sides to not drag mean towards any value
        vector<vector<double>> y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7), sample_losses(y_pred.size(), vector<double>(y_pred[0].size(), 0));
        vector<double> sample_losses_mean(y_pred.size(), 0);
        
        // Calculate sample-wise loss
        for(int i = 0; i < y_true.size(); i++){
            for(int j = 0; j < y_true[0].size(); j++)
                sample_losses[i][j] = -1 * (y_true[i][j] * log(y_pred_clipped[i][j]) + (1 - y_true[i][j]) * log(1 - y_pred_clipped[i][j]));
        }

        // This mean calculation is unnecessary because the input is (n_samples, 1) and it's taking mean rowwise which will be the same value 
        // because we have only 1 column due to binary classification we have only 1 neuron. And even in the case of multiple neuron each representing binary class
        // this calculation will make no sense, we need to calculate mean by another way in that case.
        for(int i = 0; i < y_true.size(); i++){
            double total = 0;
            for(int j = 0; j < y_true[0].size(); j++)
                total += sample_losses[i][j];
            sample_losses_mean[i] = total / y_true[0].size();
        }

        return sample_losses_mean;
    }

    void backward(vector<vector<double>>& dvalues, vector<vector<double>>& y_true){

        // Clip data to prevent division by 0
        // Clip both sides to not drag mean towards any value
        vector<vector<double>> clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7);

        // Calculate gradient
        this->dinputs.resize(dvalues.size(), vector<double>(dvalues[0].size(), 0));
        for(int i = 0; i < dvalues.size(); i++){
            for(int j = 0; j < dvalues[0].size(); j++)
                this->dinputs[i][j] = -1 * (y_true[i][j] / clipped_dvalues[i][j] - (1 - y_true[i][j]) / (1 - clipped_dvalues[i][j])) / dvalues[0].size();
        }

        for(int i = 0; i < dvalues.size(); i++){
            for(int j = 0; j < dvalues[0].size(); j++)
                this->dinputs[i][j] = this->dinputs[i][j] / dvalues.size();
        }
    }
};

#endif