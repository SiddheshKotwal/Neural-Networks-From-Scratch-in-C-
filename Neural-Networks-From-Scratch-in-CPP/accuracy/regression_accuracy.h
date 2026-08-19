#ifndef regression_accuracy
#define regression_accuracy

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

#include "calculate_accuracy.h"
#include "../math_operations/numpy_operations.h"

using namespace std;

class Accuracy_Regression : public Accuracy{
public:
    void compare(vector<vector<double>>& y_pred, vector<vector<double>>& y_true) {
        numpy np;
        double accuracy_precision = np.stddev(y_true) / 250;
        this->comparisons.resize(y_true.size(), 0);

        // Calculating accuracy in parallel
        for (int i = 0; i < y_pred.size(); i++) {
            if (abs(y_pred[i][0] - y_true[i][0]) < accuracy_precision) this->comparisons[i] = 1;
            else this->comparisons[i] = 0;
        }
    }
};

#endif // regression_accuracy
