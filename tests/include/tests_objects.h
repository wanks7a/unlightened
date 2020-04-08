#pragma once
#include <Layer.h>

struct test_layer : public Layer
{
    test_layer()
    {
        device_layer = true;
    }

    cuVector<float> output;
    void init(const shape& input) override {};
    void forward_pass(Layer* prevLayer) override {};
    void backprop(Layer* layer) override {};
    const float* get_output()  override
    {
        return output.get();
    };
    const float* derivative_wr_to_input() override
    {
        return output.get();
    };

    void set_output_shape(const shape& sh)
    {
        output_shape = sh;
    }
};
