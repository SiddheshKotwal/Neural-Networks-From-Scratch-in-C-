#ifndef mean_absolute_error
#define mean_absolute_error

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

// This function, used as a loss, penalizes the error linearly. It produces sparser results and is robust
// to outliers, which can be both advantageous and disadvantageous. In reality, L1 (MAE) loss is
// used less frequently than L2 (MSE) loss.
// MAE : L(i) = 1 / J * (Sum(j) abs(y(i, j) - y_hat(i, j)))

// We already calculated the partial derivative of an absolute value for the L1 regularization, which
// is similar to the L1 loss. The derivative of an absolute value equals 1 if this value is greater than
// 0, or -1 if it’s less than 0. The derivative does not exist for a value of 0:
// derivative of absolute value is if below 0 -> -1, if above 0 -> +1 and for 0 -> 0 

class Loss_MeanAboluteError : public Loss{
    public:
    vector<vector<double>> dinputs;

    vector<double> forward(vector<vector<double>>& y_pred, vector<vector<double>>& y_true){
        vector<double> samples_losses(y_pred.size());

        for(int i = 0; i < y_pred.size(); i++){
            for(int j = 0; j < y_pred[0].size(); j++)
                samples_losses[i] += abs(y_true[i][j] - y_pred[i][j]);
            samples_losses[i] /= y_pred[0].size(); 
        }

        return samples_losses;
    }

    void backward(vector<vector<double>>& dvalues, vector<vector<double>>& y_true){
        
        this->dinputs.resize(dvalues.size(), vector<double>(dvalues[0].size(), 0));
        for(int i = 0; i < y_true.size(); i++){
            for(int j = 0; j < y_true[0].size(); j++){
                if(y_true[i][j] - dvalues[i][j] < 0) this->dinputs[i][j] = -1 / y_true[0].size();
                else if(y_true[i][j] - dvalues[i][j] > 0) this->dinputs[i][j] = 1 / y_true[0].size();
                else this->dinputs[i][j] = 0;   // calculating the gradient
                this->dinputs[i][j] /= y_true.size();   // normalizing the gradient
            }
        }
    }
};

#endif