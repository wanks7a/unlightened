#pragma once
#include "Layer.h"
#include <vector>
#include <iostream>

class SigmoidLayer : public Layer
{
    std::vector<float> output;
    std::vector<float> derivativeWRtoInput;
    std::vector<float> verifiedGradient;
    std::vector<float> test;
    size_t size;
    size_t input_size;

    void init(const shape& input) override
    {
        input_size = input.size();
        size = input_size;
        output.resize(size);
        derivativeWRtoInput.resize(input_size);
        verifiedGradient.resize(input_size);
        test.resize(input_size);
        output_shape.width = input_size;
    }

    void forwardPass(Layer* prevLayer) override
    {
        const float* input = prevLayer->getOutput();
        for (size_t i = 0; i < size; i++)
        {
            output[i] = sigmoid(input[i]);
        }
        output[size - 1] = 1.0f;
    }

    void backprop(Layer* layer) override
    {
        const float* derivativeWRtoOutput = layer->derivativeWithRespectToInput();
        for (size_t i = 0; i < input_size; i++)
        {
            derivativeWRtoInput[i] = output[i] * (1 - output[i]) * derivativeWRtoOutput[i];
        }
    }

    const float* getOutput() override
    {
        return output.data();
    }

    const float* derivativeWithRespectToInput() override
    {
        return derivativeWRtoInput.data();
    }

    void printLayer() override
    {
        std::cout << "Sigmoid Layer" << std::endl;
        for (size_t i = 0; i < size; i++)
        {
            std::cout << "output[" << i << "] = " << output[i] << std::endl;
        }
    }

private:
    float sigmoid(float input)
    {
        input = exp((-1.0f) * input);
        return 1 / (1 + input);
    }
};
