#pragma once
#include "Layer.h"
#include <vector>
#include <iostream>

class InputLayer : public Layer
{
    std::vector<float> output;
public:
    InputLayer(shape shape)
    {
        input_shape = shape;
        output_shape = shape;
        output.resize(output_shape.size());
    }

    void init(const shape& input) override
    {
    }

    bool set_input(const float* data, size_t size)
    {
        if (size != input_shape.size())
            return false;
        memcpy(output.data(), data, size * sizeof(float));
        return true;
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
        return output.data();
    };
};