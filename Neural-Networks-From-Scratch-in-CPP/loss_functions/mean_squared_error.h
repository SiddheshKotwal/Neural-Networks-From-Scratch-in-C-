#ifndef mean_squared_error
#define mean_squared_error

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

// The two main methods for calculating error in regression are mean squared error (MSE) and mean absolute error (MAE).
// MSE : L(i) = 1 / J * (Sum(j) (y(i, j) - y_hat(i, j)) ^ 2)
// The idea here is to penalize more harshly the further away we get from the intended target
// Similarly to the other loss function implementations, we also normalize gradients by the
// number of samples to make them invariant to the batch size, or the number of samples in
// general.
 
class Loss_MeanSquaredError : public Loss{  // L2 loss
    public:
    vector<vector<double>> dinputs;

    vector<double> forward(vector<vector<double>>& y_pred, vector<vector<double>>& y_true){
        
        vector<double> sample_losses(y_true.size(), 0);
        for(int i = 0; i < y_true.size(); i++){
            for(int j = 0; j < y_true[0].size(); j++)
                sample_losses[i] += pow((y_true[i][j] - y_pred[i][j]), 2);
            sample_losses[i] /=  y_pred[0].size();
        }

        return sample_losses;
    }

    void backward(vector<vector<double>>& dvalues, vector<vector<double>>& y_true){
        
        this->dinputs.resize(dvalues.size(), vector<double>(dvalues[0].size(), 0)); // setting same size
        for(int i = 0; i < y_true.size(); i++){
            for(int j = 0; j < y_true[0].size(); j++)
                this->dinputs[i][j] = (-2 * (y_true[i][j] - dvalues[i][j]) / dvalues[0].size()) / y_true.size();    // calculating and normalizing the gradient
        }
    }
};

#endif