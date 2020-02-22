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

    void forward_pass(Layer* prevLayer) override
    {
        const float* input = prevLayer->get_output();
        for (size_t i = 0; i < size; i++)
        {
            output[i] = sigmoid(input[i]);
        }
        output[size - 1] = 1.0f;
    }

    void backprop(Layer* layer) override
    {
        const float* derivativeWRtoOutput = layer->derivative_wr_to_input();
        for (size_t i = 0; i < input_size; i++)
        {
            derivativeWRtoInput[i] = output[i] * (1 - output[i]) * derivativeWRtoOutput[i];
        }
    }

    const float* get_output() override
    {
        return output.data();
    }

    const float* derivative_wr_to_input() override
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

    ~SigmoidLayer() = default;
private:
    float sigmoid(float input)
    {
        input = exp((-1.0f) * input);
        return 1 / (1 + input);
    }
};
