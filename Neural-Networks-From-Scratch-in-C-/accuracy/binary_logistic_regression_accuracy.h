#ifndef binary_logistic_regression_accuracy
#define binary_logistic_regression_accuracy

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
using namespace std;

class Accuracy_Logistic_Regression {
public:
    double calculate(vector<vector<double>>& predictions, vector<vector<double>>& y_true) {
        vector<vector<double>> pred;
        double total = 0, accuracy_ = 0;
        pred = predictions;

        for (int i = 0; i < pred.size(); i++) {
            for (int j = 0; j < pred[0].size(); j++) {
                pred[i][j] = (predictions[i][j] > 0.5) ? 1 : 0;
                if (pred[i][j] == y_true[i][j]) total++;
            }
        }

        accuracy_ = total / (pred.size() * pred[0].size());
        return accuracy_;
    }
};

#endif // binary_logistic_regression_accuracy
