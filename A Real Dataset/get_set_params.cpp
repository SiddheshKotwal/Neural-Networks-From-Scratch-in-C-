#include "../Neural-Networks-From-Scratch-in-CPP/Layers/dense_layer.h"

#include <iostream>
#include <string>
#include <tuple>
#include <vector>

using Matrix = std::vector<std::vector<double>>;
using LayerParameters = std::tuple<Matrix, Matrix>;
using ModelParameters = std::vector<LayerParameters>;

void save_parameters(
    const ModelParameters& data,
    const std::string& filename
);

void load_parameters(
    ModelParameters& data,
    const std::string& filename
);

using namespace std;

vector<tuple<vector<vector<double>>, vector<vector<double>>>> get_parameters(vector<Layer_Dense>& layer_params){
    vector<tuple<vector<vector<double>>, vector<vector<double>>>> parameters;
    for(int i = 0; i < layer_params.size(); i++)
        parameters.push_back(layer_params[i].get_parameters());
    return parameters;
}

void set_parameters(vector<Layer_Dense*>& layer_params, vector<tuple<vector<vector<double>>, vector<vector<double>>>>& parameters){
    for(int i = 0; i < layer_params.size(); i++)
        layer_params[i]->set_parameters(parameters[i]);
}

#include "save_and_load.cpp"