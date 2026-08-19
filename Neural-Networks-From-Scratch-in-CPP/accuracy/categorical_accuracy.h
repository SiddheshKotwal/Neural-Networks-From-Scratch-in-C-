#ifndef categorical_accuracy
#define categorical_accuracy

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

class Accuracy_Categorical : public Accuracy{
public:
    // Sparse Target classes
    void compare(vector<vector<double>>& predictions, vector<double>& class_targets) {
        numpy np;
        vector<double> pred_classes(predictions.size(), 0);
        vector<double> target_classes = class_targets;
        pred_classes = np.argmax(predictions);
        this->comparisons.resize(pred_classes.size(), 0);

        // Calculating total correct predictions in parallel
        for (int i = 0; i < predictions.size(); i++) {
            if (pred_classes[i] == target_classes[i]) this->comparisons[i] = 1;
            else this->comparisons[i] = 0;
        }
    }

    // Function overloading in case of one-hot encoded class target values
    void compare(vector<vector<double>>& predictions, vector<vector<double>>& class_targets) {
        numpy np;
        vector<double> pred_classes(predictions.size(), 0), target_classes(predictions.size(), 0);
        pred_classes = np.argmax(predictions);
        target_classes = np.argmax(class_targets);
        this->comparisons.resize(pred_classes.size(), 0);

        // Calculating total correct predictions in parallel
        for (int i = 0; i < predictions.size(); i++) {
            if (pred_classes[i] == target_classes[i]) this->comparisons[i] = 1;
            else this->comparisons[i] = 0;
        }
    }
};

#endif // categorical_accuracy
