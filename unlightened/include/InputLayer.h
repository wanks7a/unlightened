#pragma once
#include "Layer.h"
#include <vector>
#include <iostream>

class InputLayer : public Layer
{
    std::vector<float> outputNeurons;
    shape input_shape;
public:
    InputLayer(size_t size, bool useBias)
    {
        input_shape.height = size;
        if (useBias)
            size = size + 1;
        outputNeurons.resize(size);
        output_shape.height = size;
        outputNeurons.back() = 1.0f;
    }

    void init(const shape& input) override
    {
    }

    bool setInput(const float* data, size_t dataSize)
    {
        if (dataSize != input_shape.size())
            return false;
        memcpy(outputNeurons.data(), data, dataSize * sizeof(float));
        return true;
    }

    void set_output_shape(shape output_sh)
    {
        output_shape = output_sh;
        input_shape = output_sh;
    }

    void forward_pass(Layer* prevLayer) override
    {
    }

    void backprop(Layer* layer) override
    {
    }

    const float* derivative_wr_to_input() override
    {
        return nullptr;
    }

    const float* get_output() override
    {
        return outputNeurons.data();
    };

    void printLayer() override
    {
        std::cout << "Input Layer" << std::endl;
        for (size_t i = 0; i < output_shape.height; i++)
        {
            std::cout << "input[" << i << "] = " << outputNeurons[i] << std::endl;
        }
    }
};