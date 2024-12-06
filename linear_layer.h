//
// Created by kevin on 06/12/24.
//

#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <vector>
#include "neuron.h"

class LinearLayer {
public:
    LinearLayer(int numNeurons);
    std::vector<float> forward(const std::vector<float>& inputs);
private:
    std::vector<Neuron> neurons;
};

#endif // LINEAR_LAYER_H

