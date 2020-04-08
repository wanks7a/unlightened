#pragma once
#include "Layer.h"
#include <vector>
#include <iostream>

class OutputLayer : public Layer
{   
    std::vector<float> predictedValue;
    std::vector<float> derivativeWRToInput;
    std::vector<float> observedValues;
    size_t size;
public:

    OutputLayer() : size(0)
    {
    }

    void init(const shape& input) override
    {
        output_shape = input;
        size = input.size();
        derivativeWRToInput.resize(size);
        observedValues.resize(size);
        output_shape.width = size;
    }

    void forward_pass(Layer* prevLayer) override
    {
        predictedValue = prevLayer->get_native_output();
    };

    void backprop(Layer* layer) override
    {
        for (size_t i = 0; i < size; i++)
        {
            derivativeWRToInput[i] = -2 * (observedValues[i] - predictedValue[i]);
        }
    }

    bool setObservedValue(const std::vector<float>& observedVal)
    {
        if (observedVal.size() == observedValues.size())
        {
            observedValues = observedVal;
            return true;
        }
        return false;
    }

    const float* get_output() override
    {
        return predictedValue.data();
    }
    const float* derivative_wr_to_input() override
    {
        return derivativeWRToInput.data();
    }

    ~OutputLayer() = default;
};
