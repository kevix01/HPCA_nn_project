//
// Created by kevin on 06/12/24.
//

#ifndef NEURON_H
#define NEURON_H

#include <vector>

class Neuron {
public:
    Neuron();
    float activate(const std::vector<float>& inputs);
private:
    std::vector<float> weights;
    float bias;
    float activationFunction(float x);
};

#endif // NEURON_H

