#pragma once
#include "Layer.h"
#include <vector>
#include <iostream>

class InputLayer : public Layer
{
    std::vector<float> outputNeurons;
public:
    InputLayer(size_t size, bool useBias)
    {
        inputSize = size;
        if (useBias)
            size = size + 1;
        outputNeurons.resize(size);
        this->size = size;
        outputNeurons.back() = 1.0f;
    }

    void init() override
    {
    }

    bool setInput(const float* data, size_t dataSize)
    {
        if (dataSize != inputSize)
            return false;
        memcpy(outputNeurons.data(), data, dataSize * sizeof(float));
        return true;
    }

    void forwardPass(Layer* prevLayer) override
    {
    }

    void backprop(Layer* layer) override
    {
    }

    const float* derivativeWithRespectToInput() override
    {
        return nullptr;
    }

    const float* getOutput() override
    {
        return outputNeurons.data();
    };

    void printLayer() override
    {
        std::cout << "Input Layer" << std::endl;
        for (size_t i = 0; i < size; i++)
        {
            std::cout << "input[" << i << "] = " << outputNeurons[i] << std::endl;
        }
    }
};