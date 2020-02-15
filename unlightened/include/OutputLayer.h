#pragma once
#include "Layer.h"
#include <vector>
#include <iostream>

class OutputLayer : public Layer
{
    const float* predictedValue = nullptr;
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

    void forwardPass(Layer* prevLayer) override
    {
        predictedValue = prevLayer->getOutput();
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

    const float* getOutput() override
    {
        return predictedValue;
    }
    const float* derivativeWithRespectToInput() override
    {
        return derivativeWRToInput.data();
    }

    void printLayer() override
    {
        std::cout << "Output Layer" << std::endl;
        for (size_t i = 0; i < size; i++)
        {
            std::cout << "output[" << i << "] = " << predictedValue[i] << std::endl;
        }
    }

};
