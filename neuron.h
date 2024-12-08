//
// Created by kevin on 06/12/24.
//

#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include "device_type.h"

class Neuron {
public:
    Neuron(DeviceType device, int numInputs);  // Add numInputs parameter
    float activate(const std::vector<float>& inputs);
    float getOutput() const;
    void setDelta(float delta);
    float getDelta() const;

private:
    float bias;
    DeviceType device;
    float output;
    float delta;
    float activationFunction(float x);
    float activationFunctionDerivative(float x);
};

#endif // NEURON_H





